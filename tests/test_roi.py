import unittest

import numpy as np

from backend.bean.subtitle_area import SubtitleArea
from backend.tools.roi import SubtitleROI


def make_coordinate_image(height: int, width: int) -> np.ndarray:
    y, x = np.indices((height, width))
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = x % 256
    image[:, :, 1] = y % 256
    image[:, :, 2] = (x + y) % 256
    return image


class SubtitleROITest(unittest.TestCase):
    def test_crop_returns_padded_coordinate_encoded_region(self) -> None:
        frame = make_coordinate_image(height=40, width=60)
        roi = SubtitleROI(
            frame.shape,
            SubtitleArea(ymin=10, ymax=20, xmin=15, xmax=30),
            padding=3,
        )

        cropped = roi.crop(frame)

        self.assertEqual(roi.bounds, (12, 7, 33, 23))
        self.assertEqual(cropped.shape, (16, 21, 3))
        self.assertEqual(cropped[0, 0].tolist(), [12, 7, 19])
        self.assertEqual(cropped[-1, -1].tolist(), [32, 22, 54])
        np.testing.assert_array_equal(cropped, frame[7:23, 12:33])

    def test_crop_clips_padding_at_frame_edges(self) -> None:
        frame = make_coordinate_image(height=40, width=60)
        roi = SubtitleROI(
            frame.shape,
            SubtitleArea(ymin=0, ymax=10, xmin=0, xmax=12),
            padding=5,
        )

        cropped = roi.crop(frame)

        self.assertEqual(roi.bounds, (0, 0, 17, 15))
        self.assertEqual(cropped.shape, (15, 17, 3))
        np.testing.assert_array_equal(cropped, frame[:15, :17])

    def test_coordinate_transforms_round_trip(self) -> None:
        frame = make_coordinate_image(height=40, width=60)
        roi = SubtitleROI(
            frame.shape,
            SubtitleArea(ymin=10, ymax=20, xmin=15, xmax=30),
            padding=3,
        )
        original_points = [(12, 7), (20, 15), (32, 22)]

        relative_points = roi.to_relative(original_points)

        self.assertEqual(relative_points, [(0, 0), (8, 8), (20, 15)])
        self.assertEqual(roi.to_original(relative_points), original_points)

    def test_subtitle_area_initializes_and_owns_roi(self) -> None:
        area = SubtitleArea(ymin=10, ymax=20, xmin=15, xmax=30)

        self.assertIsNone(area.roi)
        roi = area.initialize_roi((40, 60), padding=3)

        self.assertIs(area.roi, roi)
        self.assertEqual(area.roi.bounds, (12, 7, 33, 23))


if __name__ == "__main__":
    unittest.main()
