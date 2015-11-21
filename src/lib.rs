//! Find discrete returns in full waveform LiDAR data.
//!
//! # Why is this library called `peakbag`?
//!
//! [Peak bagging](https://en.wikipedia.org/wiki/Peak_bagging) is when you try to summit a bunch of
//! mountains, just to say you summited a bunch of mountains. While the practice of peak bagging
//! can correlate with actual appreciation for the out-of-doors and an adventuresome spirit, a peak
//! bagging attitude is neither necessary nor sufficient for a good time outside.
//!
//! Some use "peak bagger" as a derisive term for someone who likes taking a selfie on top of a
//! mountain more than just spending time outside.
//!
//! This library finds peaks in waveforms, so `peakbag` seemed as good of a name as any.

#![deny(box_pointers, fat_ptr_transmutes, missing_copy_implementations, missing_debug_implementations, missing_docs, trivial_casts, unsafe_code, unused_extern_crates, unused_import_braces, unused_qualifications, variant_size_differences)]

#[macro_use]
extern crate log;
extern crate num;

use std::f64;
use std::fmt;

use num::traits::{ToPrimitive, Unsigned};

/// Detects peaks in full waveform data.
///
/// This is a convenience method that wraps calls to `PeakDetector::new` and
/// `PeakDetector::detect_peaks`.
///
/// # Examples
///
/// ```
/// let ref data = [1u32, 2, 3, 4, 3, 2, 1];
/// let peaks = peakbag::detect_peaks(data, 3, 0, 5);
/// assert_eq!(1, peaks.len());
/// assert_eq!(4, peaks[0].amplitude);
/// assert_eq!(3, peaks[0].index);
/// ```
pub fn detect_peaks<T>(data: &[T], width: usize, floor: T, ceiling: T) -> Vec<Peak<T>>
    where T: Copy + fmt::Display + PartialOrd + ToPrimitive + Unsigned
{
    let detector = PeakDetector::new(width, floor, ceiling);
    detector.detect_peaks(data)
}

/// Configurable struct for detecting peaks.
///
/// This structure allow for fine-grained adjustment of the peak detection procedures.
#[derive(Clone, Copy, Debug)]
pub struct PeakDetector<T: Copy> {
    width: usize,
    floor: T,
    ceiling: T,
    saturation: Option<T>,
    max_kurtosis: f64,
    min_height_above_background: f64,
}

impl<T: Copy> PeakDetector<T> {
    /// Creates a new peak detector.
    ///
    /// # Examples
    ///
    /// ```
    /// use peakbag::PeakDetector;
    /// let detector = PeakDetector::new(1, 2, 3);
    /// ```
    pub fn new(width: usize, floor: T, ceiling: T) -> PeakDetector<T> {
        PeakDetector {
            width: width,
            floor: floor,
            ceiling: ceiling,
            saturation: None,
            max_kurtosis: f64::MAX,
            min_height_above_background: f64::MIN,
        }
    }

    /// Sets the saturation level for this peak detector.
    ///
    /// # Examples
    ///
    /// ```
    /// use peakbag::PeakDetector;
    /// let mut detector = PeakDetector::new(1, 2, 3).saturation(3);
    /// ```
    pub fn saturation(mut self, saturation: T) -> PeakDetector<T> {
        self.saturation = Some(saturation);
        self
    }

    /// Sets the minimum allowable height above background for a peak.
    ///
    /// # Examples
    ///
    /// ```
    /// use peakbag::PeakDetector;
    /// let mut detector = PeakDetector::new(1, 2, 3).min_height_above_background(4.0);
    /// ```
    pub fn min_height_above_background(mut self,
                                       min_height_above_background: f64)
                                       -> PeakDetector<T> {
        self.min_height_above_background = min_height_above_background;
        self
    }

    /// Sets the maximum allowable kurtosis for a peak.
    ///
    /// # Examples
    ///
    /// ```
    /// use peakbag::PeakDetector;
    /// let detector = PeakDetector::new(1, 2, 3).max_kurtosis(4.0);
    /// ```
    pub fn max_kurtosis(mut self, max_kurtosis: f64) -> PeakDetector<T> {
        self.max_kurtosis = max_kurtosis;
        self
    }
}

impl<T> PeakDetector<T> where T: Copy + fmt::Display + PartialOrd + ToPrimitive + Unsigned {
    /// Detects peaks in full waveform data.
    ///
    /// # Panics
    ///
    /// Panics if a sample value cannot be converted to an `i64` and `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use peakbag::PeakDetector;
    /// let detector = PeakDetector::new(0, 8, 2);
    /// let peaks = detector.detect_peaks(&[1u32, 2, 3, 2, 1]);
    /// ```
    pub fn detect_peaks(&self, data: &[T]) -> Vec<Peak<T>> {
        let mut state = State::Ascending(0);
        let mut peaks = Vec::new();
        for (i, &sample) in data.iter().enumerate() {
            if i == 0 {
                // We assume the first sample has a slope of zero, which means it can start a leading
                // edge.
                state = State::Ascending(1);
                continue;
            }
            let slope = sample.to_i64().unwrap() - data[i - 1].to_i64().unwrap();
            if let Some(s) = self.saturation {
                if sample == s {
                    state = State::Saturated;
                }
            }
            match state {
                State::Ascending(n) => {
                    if slope < 0 {
                        state = State::Ascending(0);
                    } else if n == self.width {
                        state = State::Descending(0, i);
                    } else {
                        state = State::Ascending(n + 1);
                    }
                }
                State::Descending(n, index) => {
                    if slope > 0 {
                        if n == 0 {
                            state = State::Descending(0, i);
                        } else {
                            state = State::Ascending(1);
                        }
                    } else if n + 1 == self.width {
                        let amplitude = data[index];
                        if amplitude > self.floor && amplitude <= self.ceiling {
                            let (mean, rms, kurtosis) = self.peak_stats(&data, index);
                            let height_above_background = self.height_above_background(&data,
                                                                                       index);
                            if height_above_background >= self.min_height_above_background &&
                               kurtosis < self.max_kurtosis {
                                peaks.push(Peak {
                                    index: index,
                                    amplitude: amplitude,
                                    mean: mean,
                                    rms: rms,
                                    kurtosis: kurtosis,
                                    height_above_background: height_above_background,
                                });
                            }
                        }
                        state = State::Ascending(0);
                    } else {
                        state = State::Descending(n + 1, index);
                    }
                }
                State::Saturated => {
                    // We know if we're below the floor, we have a negative slope, since we must
                    // have just gone below the floor.
                    if sample <= self.floor {
                        state = State::Ascending(0);
                    }
                }
            };
            debug!("({}) sample={}, slope={}, state={:?}",
                   i,
                   sample,
                   slope,
                   state);
        }
        peaks
    }

    fn peak_stats(&self, data: &[T], index: usize) -> (f64, f64, f64) {
        let mut values = 0u64;
        let mut values2 = 0u64;
        let mut nvalues = 0usize;
        for &sample in data.iter().skip(index - self.width).take(self.width * 2 + 1) {
            let sample = sample.to_u64().unwrap();
            values += sample;
            values2 += sample * sample;
            nvalues += 1;
        }
        let mean = values as f64 / nvalues as f64;
        let rms = (values2 as f64 / nvalues as f64 - (values as f64 / nvalues as f64).powi(2))
                      .sqrt();
        let mut kurtosis = 0f64;
        for &sample in data.iter().skip(index - self.width).take(self.width * 2 + 1) {
            let sample = sample.to_u64().unwrap();
            let temp = (sample as f64 - mean) / rms;
            kurtosis += temp.powi(4);
        }
        kurtosis = kurtosis / nvalues as f64 - 3.0;
        (mean, rms, kurtosis)
    }

    fn height_above_background(&self, data: &[T], index: usize) -> f64 {
        let slope: f64 = (data[index + self.width] - data[index - self.width]).to_f64().unwrap() /
                         (2.0 * self.width as f64);
        let intercept = data[index + self.width].to_f64().unwrap() -
                        slope * (index - self.width) as f64;
        data[index].to_f64().unwrap() - (slope * index as f64 + intercept)
    }
}

#[derive(Debug)]
enum State {
    Ascending(usize),
    Descending(usize, usize),
    Saturated,
}

/// A peak in the waveform data.
#[derive(Clone, Copy, Debug)]
pub struct Peak<T: Copy> {
    /// The raw intensity of the peak.
    pub amplitude: T,
    /// The index of the peak in the sample data.
    pub index: usize,
    /// The mean intensity value of the peak.
    pub mean: f64,
    /// The rms error of the peak from that mean.
    pub rms: f64,
    /// The kurtosis of the peak.
    pub kurtosis: f64,
    /// The height of the peak above a background level, as defined by the first and last points in
    /// the peak.
    pub height_above_background: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_peak() {
        let peaks = detect_peaks(&[1u32, 2, 3, 4, 4, 4, 4], 3, 0, 5);
        assert_eq!(1, peaks.len());
        assert_eq!(3, peaks[0].index);
    }

    #[test]
    fn floor() {
        // Must be above floor, below or equal to ceil
        let peaks = detect_peaks(&[1u32, 2, 3, 4, 3, 2, 1], 3, 4, 5);
        assert_eq!(0, peaks.len());
    }

    #[test]
    fn ceiling() {
        // Must be above floor, below or equal to ceil
        let peaks = detect_peaks(&[1u32, 2, 3, 4, 3, 2, 1], 3, 0, 3);
        assert_eq!(0, peaks.len());
    }

    #[test]
    fn saturation() {
        let detector = PeakDetector::new(3, 1, 8).saturation(8);
        let peaks = detector.detect_peaks(&[5u32, 6, 7, 8, 7, 6, 5]);
        assert_eq!(0, peaks.len());
    }

    #[test]
    fn stats() {
        let peaks = detect_peaks(&[1u32, 2, 3, 4, 3, 2, 1], 3, 0, 5);
        let ref peak = peaks[0];
        assert_eq!(2.2857142857142856, peak.mean);
        assert_eq!(1.0301575072754257, peak.rms);
        assert_eq!(-1.143491124260356, peak.kurtosis);
        assert_eq!(3.0, peak.height_above_background);
    }

    #[test]
    fn min_height_above_background() {
        let detector = PeakDetector::new(3, 1, 8).min_height_above_background(4.0);
        let peaks = detector.detect_peaks(&[1u32, 2, 3, 4, 3, 2, 1]);
        assert_eq!(0, peaks.len());
    }

    #[test]
    fn peak_kurtosis() {
        let detector = PeakDetector::new(3, 1, 8).max_kurtosis(-2.0);
        let peaks = detector.detect_peaks(&[1u32, 2, 3, 4, 3, 2, 1]);
        assert_eq!(0, peaks.len());
    }

    #[test]
    fn should_be_one_peak() {
        let ref data = [2u16, 2, 1, 2, 5, 18, 51, 107, 166, 195, 176, 125, 70, 34, 14, 7, 5, 4, 5,
                        4, 3, 1, 0, 0];
        let peaks = detect_peaks(data, 2, 15, 255);
        assert_eq!(1, peaks.len());
    }
}
