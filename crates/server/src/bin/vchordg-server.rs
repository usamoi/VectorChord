use byteorder::{LE, ReadBytesExt};
use index::accessor::L2S;
use index::fetch::Fetch;
use index::prefetcher::{PlainPrefetcher, PrefetcherSequenceFamily, Sequence};
use index::relation::RelationRead;
use server::api::{ClientPacket, ServerPacket};
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::TcpStream;
use std::num::NonZero;
use simd::f16;
use vchordg::operator::Op;
use vchordg::types::OwnedVector;
use vector::VectorOwned;
use vector::vect::{VectBorrowed, VectOwned};

fn server_loop(connection: &TcpStream) -> anyhow::Result<()> {
    let mut reader = BufReader::new(connection);
    let mut buf = Vec::<u8>::new();
    let index = server::mock::MockRelation::<vchordg::Opaque>::new(1800000);
    let vector_options;
    let index_options;
    {
        let length = reader.read_u64::<LE>()?;
        buf.resize(length as _, 0);
        reader.read_exact(buf.as_mut_slice())?;
        let packet: ClientPacket = serde_json::from_slice(&buf)?;
        if let ClientPacket::Build {
            vector_options: vec_ops,
            index_options: idx_ops,
        } = packet
        {
            vector_options = vec_ops;
            index_options = idx_ops;
        } else {
            anyhow::bail!("first packet is not build");
        }
    }
    match (vector_options.v, vector_options.d) {
        (vchordg::types::VectorKind::Vecf32, vchordg::types::DistanceKind::L2S) => {
            vchordg::build::<_, Op<VectOwned<f32>, L2S>>(
                vector_options.clone(),
                index_options,
                &index,
            );
        }
        (vchordg::types::VectorKind::Vecf32, vchordg::types::DistanceKind::Dot) => todo!(),
        (vchordg::types::VectorKind::Vecf16, vchordg::types::DistanceKind::L2S) => todo!(),
        (vchordg::types::VectorKind::Vecf16, vchordg::types::DistanceKind::Dot) => todo!(),
        (vchordg::types::VectorKind::Rabitq8, vchordg::types::DistanceKind::L2S) => todo!(),
        (vchordg::types::VectorKind::Rabitq8, vchordg::types::DistanceKind::Dot) => todo!(),
        (vchordg::types::VectorKind::Rabitq4, vchordg::types::DistanceKind::L2S) => todo!(),
        (vchordg::types::VectorKind::Rabitq4, vchordg::types::DistanceKind::Dot) => todo!(),
    }
    {
        let (tx, rx) = crossbeam_channel::bounded::<(OwnedVector, NonZero<u64>)>(1024);
        std::thread::scope(|scope| -> anyhow::Result<_> {
            let mut threads = Vec::new();
            for _ in 0..20 {
                threads.push(scope.spawn({
                    let index = index.clone();
                    let vector_options = vector_options.clone();
                    let rx = rx.clone();
                    move || {
                        while let Ok((vector, payload)) = rx.recv() {
                            let bump = bumpalo::Bump::new();
                            match (vector_options.v, vector_options.d) {
                                (
                                    vchordg::types::VectorKind::Vecf32,
                                    vchordg::types::DistanceKind::L2S,
                                ) => {
                                    let OwnedVector::Vecf32(unprojected) = vector else {
                                        unreachable!()
                                    };
                                    let projected =
                                        RandomProject::project(unprojected.as_borrowed());
                                    vchordg::insert::<_, Op<VectOwned<f32>, L2S>>(
                                        &index,
                                        projected.as_borrowed(),
                                        payload,
                                        &bump,
                                        MakePlainPrefetcher { index: &index },
                                        MakePlainPrefetcher { index: &index },
                                    );
                                }
                                (
                                    vchordg::types::VectorKind::Vecf32,
                                    vchordg::types::DistanceKind::Dot,
                                ) => todo!(),
                                (
                                    vchordg::types::VectorKind::Vecf16,
                                    vchordg::types::DistanceKind::L2S,
                                ) => todo!(),
                                (
                                    vchordg::types::VectorKind::Vecf16,
                                    vchordg::types::DistanceKind::Dot,
                                ) => todo!(),
                                (
                                    vchordg::types::VectorKind::Rabitq8,
                                    vchordg::types::DistanceKind::L2S,
                                ) => todo!(),
                                (
                                    vchordg::types::VectorKind::Rabitq8,
                                    vchordg::types::DistanceKind::Dot,
                                ) => todo!(),
                                (
                                    vchordg::types::VectorKind::Rabitq4,
                                    vchordg::types::DistanceKind::L2S,
                                ) => todo!(),
                                (
                                    vchordg::types::VectorKind::Rabitq4,
                                    vchordg::types::DistanceKind::Dot,
                                ) => todo!(),
                            }
                        }
                        eprintln!("Worker thread exited.");
                    }
                }));
            }
            drop(rx);
            loop {
                let length = reader.read_u64::<LE>()?;
                buf.resize(length as _, 0);
                reader.read_exact(buf.as_mut_slice())?;
                let packet: ClientPacket = serde_json::from_slice(&buf)?;
                match packet {
                    ClientPacket::Insert { vector, payload } => {
                        tx.send((vector, payload))?;
                    }
                    ClientPacket::Finish {} => break,
                    _ => anyhow::bail!("packet is not insert or finish"),
                }
            }
            drop(tx);
            for thread in threads {
                thread.join().expect("failed");
            }
            Ok(())
        })?;
    }
    let mut writer = BufWriter::new(connection);
    let n = index.registry.n.load(std::sync::atomic::Ordering::SeqCst);
    for id in 0..n {
        let guard = index.read(id);
        let msg = serde_json::to_string(&ServerPacket::Flush {
            id,
            pd_lower: guard.header.pd_lower,
            pd_upper: guard.header.pd_upper,
            pd_special: guard.header.pd_special,
            content: guard.content.to_vec(),
        })?;
        writer.write_all(&msg.len().to_le_bytes())?;
        writer.write_all(msg.as_bytes())?;
    }
    {
        let msg = serde_json::to_string(&ServerPacket::Finish {})?;
        writer.write_all(&msg.len().to_le_bytes())?;
        writer.write_all(msg.as_bytes())?;
    }
    Ok(())
}

pub trait RandomProject {
    type Output;
    fn project(self) -> Self::Output;
}

impl RandomProject for VectBorrowed<'_, f32> {
    type Output = VectOwned<f32>;
    fn project(self) -> VectOwned<f32> {
        use rabitq::rotate::rotate;
        let input = self.slice();
        VectOwned::new(rotate(input))
    }
}

impl RandomProject for VectBorrowed<'_, f16> {
    type Output = VectOwned<f16>;
    fn project(self) -> VectOwned<f16> {
        use rabitq::rotate::rotate;
        use simd::Floating;
        let input = f16::vector_to_f32(self.slice());
        VectOwned::new(f16::vector_from_f32(&rotate(&input)))
    }
}

#[derive(Debug)]
pub struct MakePlainPrefetcher<'b, R> {
    pub index: &'b R,
}

impl<'b, R> Clone for MakePlainPrefetcher<'b, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'b, R: RelationRead> PrefetcherSequenceFamily<'b, R> for MakePlainPrefetcher<'b, R> {
    type P<S: Sequence>
        = PlainPrefetcher<'b, R, S>
    where
        S::Item: Fetch<'b>;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch<'b>,
    {
        PlainPrefetcher::new(self.index, seq)
    }

    fn is_not_plain(&self) -> bool {
        false
    }
}

fn main() -> anyhow::Result<()> {
    let listener = std::net::TcpListener::bind("0.0.0.0:9999")?;
    loop {
        let (mut connection, _) = listener.accept()?;
        let _ = std::thread::spawn(move || {
            if let Err(error) = server_loop(&connection) {
                let msg = serde_json::to_string(&ServerPacket::Error {
                    reason: error.to_string(),
                })
                .expect("failed to serialize");
                connection
                    .write_all(&msg.len().to_le_bytes())
                    .expect("failed to write");
                connection
                    .write_all(msg.as_bytes())
                    .expect("failed to write");
            }
        });
    }
}
