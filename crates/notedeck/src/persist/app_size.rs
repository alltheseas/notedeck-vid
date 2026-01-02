use std::time::Duration;

use egui::Context;

use crate::timed_serializer::TimedSerializer;
use crate::{DataPath, DataPathType};

pub struct AppSizeHandler {
    serializer: TimedSerializer<egui::Vec2>,
}

impl AppSizeHandler {
    pub fn new(path: &DataPath) -> Self {
        let serializer =
            TimedSerializer::new(path, DataPathType::Setting, "app_size.json".to_owned())
                .with_delay(Duration::from_millis(500));

        Self { serializer }
    }

    /// Attempts to save the current application window size to the configured timed serializer.
    ///
    /// The current screen size is read from the provided `egui::Context` and passed to the internal
    /// `TimedSerializer`. The serializer applies its configured delay to avoid frequent IO while the
    /// window is being resized.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use your_crate::{AppSizeHandler, DataPath};
    /// use egui::Context;
    ///
    /// // `data_path` and `ctx` should be created according to your application's setup.
    /// let data_path = DataPath::new("settings");
    /// let mut handler = AppSizeHandler::new(&data_path);
    /// // `ctx` would be the egui context you have access to in your app
    /// let ctx: Context = /* obtain egui::Context from your app */ unimplemented!();
    /// handler.try_save_app_size(&ctx);
    /// ```
    pub fn try_save_app_size(&mut self, ctx: &Context) {
        // There doesn't seem to be a way to check if user is resizing window, so if the rect is different than last saved, we'll wait DELAY before saving again to avoid spamming io
        let cur_size = ctx.input(|i| i.screen_rect().size());
        self.serializer.try_save(cur_size);
    }

    pub fn get_app_size(&self) -> Option<egui::Vec2> {
        self.serializer.get_item()
    }
}