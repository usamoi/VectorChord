mod kernel;

use anyhow::Context;
use std::cell::OnceCell;
use std::sync::OnceLock;

pub fn main() {
    use kernel::{Client, Server};
    use std::io::{BufReader, BufWriter, Write};
    use vchordrq::cuda::rpc;
    let pwd = std::env::current_dir().expect("failed to access current directory");
    let metadata = std::fs::metadata(pwd).expect("failed to get metadata");
    #[cfg(target_family = "unix")]
    {
        use std::fs::Permissions;
        use std::os::unix::fs::PermissionsExt;
        let permissions = metadata.permissions();
        assert_eq!(permissions, Permissions::from_mode(0o700));
    }
    let server = OnceLock::new();
    let listener = std::os::unix::net::UnixListener::bind(".s").expect("failed to create listener");
    println!("Ready.");
    std::thread::scope(|scope| {
        loop {
            let (stream, _) = listener.accept().expect("failed to create connection");
            let server = &server;
            scope.spawn(move || {
                let mut client = OnceCell::new();
                let mut reader =
                    BufReader::new(stream.try_clone().expect("failed to clone stream"));
                let mut writer =
                    BufWriter::new(stream.try_clone().expect("failed to clone stream"));
                loop {
                    let request: rpc::Request =
                        bincode::decode_from_reader(&mut reader, rpc::CONFIG)
                            .expect("failed to parse query");
                    match request {
                        rpc::Request::Init(r) => {
                            let response: anyhow::Result<()> = (|| {
                                let instance = Server::new(r.op, r.d, r.n, &r.centroids)
                                    .context("failed to start server")?;
                                server
                                    .set(instance)
                                    .ok()
                                    .context("server is already initialized")?;
                                Ok(())
                            })();
                            let response = match response {
                                Ok(()) => rpc::InitResponse::Ok {},
                                Err(error) => rpc::InitResponse::Err {
                                    msg: error.to_string(),
                                },
                            };
                            bincode::encode_into_std_write(response, &mut writer, rpc::CONFIG)
                                .expect("failed to encode response");
                            writer.flush().expect("failed to flush writer");
                        }
                        rpc::Request::Connect(r) => {
                            let response: anyhow::Result<()> = (|| {
                                let server = server.get().context("server is not initialized")?;
                                let instance =
                                    Client::new(server, r.m).context("failed to start server")?;
                                client
                                    .set(instance)
                                    .ok()
                                    .context("server is already initialized")?;
                                Ok(())
                            })();
                            let response = match response {
                                Ok(()) => rpc::ConnectResponse::Ok {},
                                Err(error) => rpc::ConnectResponse::Err {
                                    msg: error.to_string(),
                                },
                            };
                            bincode::encode_into_std_write(response, &mut writer, rpc::CONFIG)
                                .expect("failed to encode response");
                            writer.flush().expect("failed to flush writer");
                        }
                        rpc::Request::Query(r) => {
                            let response: anyhow::Result<Vec<u32>> = (|| {
                                let client =
                                    client.get_mut().context("client is not initialized")?;
                                let result =
                                    client.query(r.k, &r.vectors).expect("failed to query");
                                Ok(result.to_vec())
                            })(
                            );
                            let response = match response {
                                Ok(result) => rpc::QueryResponse::Ok {
                                    result: result.to_vec(),
                                },
                                Err(error) => rpc::QueryResponse::Err {
                                    msg: error.to_string(),
                                },
                            };
                            bincode::encode_into_std_write(response, &mut writer, rpc::CONFIG)
                                .expect("failed to encode");
                            writer.flush().expect("failed to flush");
                        }
                    }
                }
            });
        }
    });
}
