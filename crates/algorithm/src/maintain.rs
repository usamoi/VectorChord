// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

use crate::closure_lifetime_binder::{id_0, id_1, id_2, id_3};
use crate::operator::{FunctionalAccessor, Operator, Vector};
use crate::tape::{self, TapeWriter, by_directory, by_next};
use crate::tape_writer::{DirectoryTapeWriter, FrozenTapeWriter};
use crate::tuples::*;
use crate::*;
use rabitq::packing::unpack;

pub fn maintain<'r, R: RelationRead + RelationWrite, O: Operator>(
    index: &'r R,
    mut prefetch_h0_tuples: impl PrefetcherSequenceFamily<'r, R>,
    check: impl Fn(),
) {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let height_of_root = meta_tuple.height_of_root();
    let freepage_first = meta_tuple.freepage_first();

    type State = Vec<u32>;
    let mut state: State = vec![meta_tuple.first()];

    drop(meta_guard);

    let step = |state: State| {
        let mut results = Vec::new();
        for first in state {
            tape::read_h1_tape::<R, _, _>(
                by_next(index, first).inspect(|_| check()),
                || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); 32])),
                |(), _, first, _| results.push(first),
            );
        }
        results
    };

    for _ in (1..height_of_root).rev() {
        state = step(state);
    }

    for first in state {
        let mut jump_guard = index.write(first, false);
        let jump_bytes = jump_guard.get_mut(1).expect("data corruption");
        let mut jump_tuple = JumpTuple::deserialize_mut(jump_bytes);

        let frozen_tape_hooked_index = RelationHooked(index, {
            id_3(move |index: &R, tracking_freespace: bool| {
                if !tracking_freespace {
                    if let Some(id) = freepages::fetch(index, freepage_first) {
                        let mut write = index.write(id, false);
                        write.clear();
                        write
                    } else {
                        index.extend(false)
                    }
                } else {
                    index.extend(true)
                }
            })
        });

        let mut tape = FrozenTapeWriter::create(
            &frozen_tape_hooked_index,
            O::Vector::count(dims as _),
            false,
        );

        let mut trace_directory = Vec::new();
        let mut trace_forzen = Vec::new();
        let mut trace_appendable = Vec::new();

        let mut tuples = 0_u64;
        let mut callback = id_2(|(code, delta): (_, _), head, payload, prefetch: &[_]| {
            tape.push(Branch {
                code,
                delta,
                prefetch: prefetch.to_vec(),
                head,
                extra: payload,
            });
            tuples += 1;
        });
        let directory = tape::read_directory_tape::<R>(
            by_next(index, *jump_tuple.directory_first())
                .inspect(|_| check())
                .inspect(|guard| trace_directory.push(guard.id())),
        );
        tape::read_frozen_tape::<R, _, _>(
            by_directory(&mut prefetch_h0_tuples, directory)
                .inspect(|_| check())
                .inspect(|guard| trace_forzen.push(guard.id())),
            || {
                FunctionalAccessor::new(
                    Vec::<[u8; 16]>::new(),
                    Vec::<[u8; 16]>::extend_from_slice,
                    id_1(|elements: Vec<_>, (code, delta): ((&[f32; 32], &[f32; 32], &[f32; 32], &[f32; 32]), _)| {
                        let unpacked = unpack(&elements);
                        std::array::from_fn(|i| {
                            let f = |&x| [x & 1 != 0, x & 2 != 0, x & 4 != 0, x & 8 != 0];
                            let signs = unpacked[i].iter().flat_map(f).collect::<Vec<_>>();
                            (rabitq::Code {dis_u_2: code.0[i], factor_ppc: code.1[i], factor_ip: code.2[i], factor_err: code.3[i], signs}, delta[i])
                        })
                    }),
                )
            },
            &mut callback,
        );
        tape::read_appendable_tape::<R, _>(
            by_next(index, *jump_tuple.appendable_first())
                .inspect(|_| check())
                .inspect(|guard| trace_appendable.push(guard.id())),
            |code, delta| {
                let signs = code
                    .1
                    .iter()
                    .flat_map(|x| std::array::from_fn::<_, 64, _>(|i| *x & (1 << i) != 0))
                    .take(dims as _)
                    .collect::<Vec<_>>();
                (
                    rabitq::Code {
                        dis_u_2: code.0.0,
                        factor_ppc: code.0.1,
                        factor_ip: code.0.2,
                        factor_err: code.0.3,
                        signs,
                    },
                    delta,
                )
            },
            &mut callback,
        );

        let (frozen_tape, branches) = tape.into_inner();

        let hooked_index = RelationHooked(
            index,
            id_3(move |index: &R, tracking_freespace: bool| {
                if !tracking_freespace {
                    if let Some(id) = freepages::fetch(index, freepage_first) {
                        let mut write = index.write(id, false);
                        write.clear();
                        write
                    } else {
                        index.extend(false)
                    }
                } else {
                    index.extend(true)
                }
            }),
        );
        let mut appendable_tape = TapeWriter::create(&hooked_index, false);

        for branch in branches {
            appendable_tape.push(AppendableTuple {
                dis_u_2: branch.code.dis_u_2,
                factor_ppc: branch.code.factor_ppc,
                factor_ip: branch.code.factor_ip,
                factor_err: branch.code.factor_err,
                elements: rabitq::pack_to_u64(&branch.code.signs),
                delta: branch.delta,
                prefetch: branch.prefetch,
                head: branch.head,
                payload: Some(branch.extra),
            });
        }

        let frozen_first = { frozen_tape }.first();

        let directory = by_next(index, frozen_first)
            .inspect(|_| check())
            .map(|guard| guard.id())
            .collect::<Vec<_>>();

        let mut directory_tape = DirectoryTapeWriter::create(index, false);
        directory_tape.push(directory.as_slice());
        let directory_tape = directory_tape.into_inner();

        *jump_tuple.directory_first() = { directory_tape }.first();
        *jump_tuple.appendable_first() = { appendable_tape }.first();
        *jump_tuple.tuples() = tuples;

        drop(jump_guard);

        let trace = {
            let mut v = Vec::new();
            v.extend_from_slice(&trace_directory);
            v.extend_from_slice(&trace_forzen);
            v.extend_from_slice(&trace_appendable);
            v
        };

        freepages::mark(index, freepage_first, trace.as_slice());
    }
}

#[derive(Clone)]
struct RelationHooked<'r, R, E>(&'r R, E);

impl<'r, R, E> Relation for RelationHooked<'r, R, E>
where
    R: Relation,
    E: Clone,
{
    type Page = R::Page;
}

impl<'r, R, E> RelationRead for RelationHooked<'r, R, E>
where
    R: RelationRead,
    E: Clone,
{
    type ReadGuard<'a>
        = R::ReadGuard<'a>
    where
        Self: 'a;

    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        self.0.read(id)
    }
}

impl<'r, R, E> RelationWrite for RelationHooked<'r, R, E>
where
    R: RelationWrite,
    E: Clone + for<'a> Fn(&'a R, bool) -> R::WriteGuard<'a>,
{
    type WriteGuard<'a>
        = R::WriteGuard<'a>
    where
        Self: 'a;

    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        self.0.write(id, tracking_freespace)
    }

    fn extend(&self, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        (self.1)(self.0, tracking_freespace)
    }

    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>> {
        self.0.search(freespace)
    }
}
