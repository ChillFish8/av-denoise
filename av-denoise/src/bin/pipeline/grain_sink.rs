use std::thread;

use av_denoise::SceneGrain;

use super::avds_file::AvdsStream;

/// Takes each scene's grain as a worker finishes it, while the run goes on.
pub struct GrainSink {
    /// The end workers send finished scenes on.
    pub sender: crossbeam_channel::Sender<SceneGrain>,
    handle: thread::JoinHandle<Result<AvdsStream, anyhow::Error>>,
}

impl GrainSink {
    /// Starts the sink thread, which writes each scene to `avds`.
    pub fn spawn(avds: AvdsStream) -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded::<SceneGrain>();
        let handle = thread::spawn(move || run_sink(&receiver, avds));

        Self { sender, handle }
    }

    /// Waits for the sink to write every scene sent before the last sender dropped.
    ///
    /// Returns the grain data file, every scene written and not yet finished.
    pub fn join(self) -> Result<AvdsStream, anyhow::Error> {
        drop(self.sender);

        self.handle
            .join()
            .map_err(|panic| anyhow::anyhow!("grain writer panicked: {panic:?}"))
            .and_then(|result| result)
    }
}

/// Writes each scene until every sender has hung up.
///
/// Returns on the first failed write, which drops the receiver so the workers fail their next send.
fn run_sink(
    receiver: &crossbeam_channel::Receiver<SceneGrain>,
    mut avds: AvdsStream,
) -> Result<AvdsStream, anyhow::Error> {
    for scene in receiver {
        avds.push(scene)?;
    }

    Ok(avds)
}
