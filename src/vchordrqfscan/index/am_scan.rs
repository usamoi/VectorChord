use super::am_options::Opfamily;
use crate::postgres::PostgresRelation;
use crate::vchordrqfscan::algorithm::scan::scan;
use crate::vchordrqfscan::gucs::executing::epsilon;
use crate::vchordrqfscan::gucs::executing::max_scan_tuples;
use crate::vchordrqfscan::gucs::executing::probes;
use crate::vchordrqfscan::types::OwnedVector;
use distance::Distance;
use std::num::NonZeroU64;

pub enum Scanner {
    Initial {
        vector: Option<(OwnedVector, Opfamily)>,
        threshold: Option<f32>,
        recheck: bool,
    },
    Vbase {
        vbase: Box<dyn Iterator<Item = (Distance, NonZeroU64)>>,
        threshold: Option<f32>,
        recheck: bool,
        opfamily: Opfamily,
    },
    Empty {},
}

pub fn scan_build(
    orderbys: Vec<Option<OwnedVector>>,
    spheres: Vec<(Option<OwnedVector>, Option<f32>)>,
    opfamily: Opfamily,
) -> (Option<(OwnedVector, Opfamily)>, Option<f32>, bool) {
    let mut pair = None;
    let mut threshold = None;
    let mut recheck = false;
    for orderby_vector in orderbys {
        if pair.is_none() {
            pair = orderby_vector;
        } else if orderby_vector.is_some() {
            pgrx::error!("vector search with multiple vectors is not supported");
        }
    }
    for (sphere_vector, sphere_threshold) in spheres {
        if pair.is_none() {
            pair = sphere_vector;
            threshold = sphere_threshold;
        } else {
            recheck = true;
            break;
        }
    }
    (pair.map(|x| (x, opfamily)), threshold, recheck)
}

pub fn scan_make(
    vector: Option<(OwnedVector, Opfamily)>,
    threshold: Option<f32>,
    recheck: bool,
) -> Scanner {
    Scanner::Initial {
        vector,
        threshold,
        recheck,
    }
}

pub fn scan_next(scanner: &mut Scanner, relation: PostgresRelation) -> Option<(NonZeroU64, bool)> {
    if let Scanner::Initial {
        vector,
        threshold,
        recheck,
    } = scanner
    {
        if let Some((vector, opfamily)) = vector.as_ref() {
            let vbase = scan(
                relation,
                match vector {
                    OwnedVector::Vecf32(x) => x.slice().to_vec(),
                },
                opfamily.distance_kind(),
                probes(),
                epsilon(),
            );
            *scanner = Scanner::Vbase {
                vbase: if let Some(max_scan_tuples) = max_scan_tuples() {
                    Box::new(vbase.take(max_scan_tuples as usize))
                } else {
                    Box::new(vbase)
                },
                threshold: *threshold,
                recheck: *recheck,
                opfamily: *opfamily,
            };
        } else {
            *scanner = Scanner::Empty {};
        }
    }
    match scanner {
        Scanner::Initial { .. } => unreachable!(),
        Scanner::Vbase {
            vbase,
            threshold,
            recheck,
            opfamily,
        } => match (
            vbase.next().map(|(d, p)| (opfamily.process(d), p)),
            threshold,
        ) {
            (Some((_, ptr)), None) => Some((ptr, *recheck)),
            (Some((distance, ptr)), Some(t)) if distance < *t => Some((ptr, *recheck)),
            _ => {
                let scanner = std::mem::replace(scanner, Scanner::Empty {});
                scan_release(scanner);
                None
            }
        },
        Scanner::Empty {} => None,
    }
}

pub fn scan_release(scanner: Scanner) {
    match scanner {
        Scanner::Initial { .. } => {}
        Scanner::Vbase { .. } => {}
        Scanner::Empty {} => {}
    }
}
