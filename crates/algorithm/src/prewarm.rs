use crate::closure_lifetime_binder::{id_0, id_1, id_2};
use crate::operator::{FunctionalAccessor, Operator};
use crate::tuples::*;
use crate::{Page, RelationRead, tape, vectors};
use std::error::Error;
use std::fmt::Write;

pub fn prewarm<R: RelationRead, O: Operator>(
    index: R,
    height: i32,
    check: impl Fn(),
) -> Result<String, Box<dyn Error>> {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let height_of_root = meta_tuple.height_of_root();
    let root_prefetch = meta_tuple.root_prefetch().to_vec();
    let root_head = meta_tuple.root_head();
    let root_first = meta_tuple.root_first();
    drop(meta_guard);

    let mut message = String::new();
    writeln!(message, "height of root: {}", height_of_root)?;
    let prewarm_max_height = if height < 0 { 0 } else { height as u32 };
    if prewarm_max_height > height_of_root {
        return Ok(message);
    }
    type State = Vec<u32>;
    let mut state: State = {
        let mut results = Vec::new();
        {
            let list = root_prefetch.into_iter().map(|id| index.read(id));
            vectors::read_for_h1_tuple::<R, O, _>(root_head, list, ());
            results.push(root_first);
        }
        writeln!(message, "------------------------")?;
        writeln!(message, "number of nodes: {}", results.len())?;
        writeln!(message, "number of pages: {}", 1)?;
        results
    };
    let mut step = |state: State| -> Result<_, Box<dyn Error>> {
        let mut counter = 0_usize;
        let mut results = Vec::new();
        for first in state {
            tape::read_h1_tape(
                index.clone(),
                first,
                || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); 32])),
                |(), head, first, prefetch| {
                    let list = prefetch.iter().map(|&id| index.read(id));
                    vectors::read_for_h1_tuple::<R, O, _>(head, list, ());
                    results.push(first);
                },
                |_| {
                    check();
                    counter += 1;
                },
            );
        }
        writeln!(message, "------------------------")?;
        writeln!(message, "number of nodes: {}", results.len())?;
        writeln!(message, "number of pages: {}", counter)?;
        Ok(results)
    };
    for _ in (std::cmp::max(1, prewarm_max_height)..height_of_root).rev() {
        state = step(state)?;
    }
    if prewarm_max_height == 0 {
        let mut counter = 0_usize;
        let mut results = Vec::new();
        for first in state {
            let jump_guard = index.read(first);
            let jump_bytes = jump_guard.get(1).expect("data corruption");
            let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
            tape::read_frozen_tape(
                index.clone(),
                jump_tuple.frozen_first(),
                || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); 32])),
                id_2(|_, _, _, _| {
                    results.push(());
                }),
                |_| {
                    check();
                    counter += 1;
                },
            );
            tape::read_appendable_tape(
                index.clone(),
                jump_tuple.appendable_first(),
                |_| (),
                id_2(|_, _, _, _| {
                    results.push(());
                }),
                |_| {
                    check();
                    counter += 1;
                },
            );
        }
        writeln!(message, "------------------------")?;
        writeln!(message, "number of nodes: {}", results.len())?;
        writeln!(message, "number of pages: {}", counter)?;
    }
    Ok(message)
}
