use std::thread;

use av_denoise::SceneGrain;

use super::avds_file::AvdsStream;

/// Takes each scene's grain as a worker finishes it, while the run goes on.
pub struct GrainSink {
    /// The end workers send finished scenes on.
    pub sender: crossbeam_channel::Sender<SceneGrain>,
    handle: thread::JoinHandle<Result<SinkOutput, anyhow::Error>>,
}

/// What the sink holds once every worker has hung up.
pub struct SinkOutput {
    /// The grain data file, every scene written and not yet finished.
    pub avds: Option<AvdsStream>,
    /// Every scene, kept only when the run asked for them.
    pub scenes: Vec<SceneGrain>,
}

impl SinkOutput {
    /// The output of a run that kept no grain.
    pub fn empty() -> Self {
        Self {
            avds: None,
            scenes: Vec::new(),
        }
    }
}

impl GrainSink {
    /// Starts the sink thread, which writes scenes to `avds` and keeps them with `keeps_scenes`.
    pub fn spawn(avds: Option<AvdsStream>, keeps_scenes: bool) -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded::<SceneGrain>();
        let handle = thread::spawn(move || run_sink(&receiver, avds, keeps_scenes));

        Self { sender, handle }
    }

    /// Waits for the sink to take every scene sent before the last sender dropped.
    pub fn join(self) -> Result<SinkOutput, anyhow::Error> {
        drop(self.sender);

        self.handle
            .join()
            .map_err(|panic| anyhow::anyhow!("grain writer panicked: {panic:?}"))
            .and_then(|result| result)
    }
}

/// Writes or keeps each scene until every sender has hung up.
///
/// Returns on the first failed write, which drops the receiver so the workers fail their next send.
fn run_sink(
    receiver: &crossbeam_channel::Receiver<SceneGrain>,
    mut avds: Option<AvdsStream>,
    keeps_scenes: bool,
) -> Result<SinkOutput, anyhow::Error> {
    let mut scenes = Vec::new();

    for scene in receiver {
        let Some(stream) = avds.as_mut() else {
            scenes.push(scene);
            continue;
        };

        if keeps_scenes {
            let kept = scene.clone();
            scenes.push(kept);
        }

        stream.push(scene)?;
    }

    Ok(SinkOutput { avds, scenes })
}
