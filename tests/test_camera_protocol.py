"""Safety contracts for browser-camera frames received over WebSocket."""

from __future__ import annotations

import base64
import unittest

import cv2
import numpy as np

from backend.main import MAX_CAMERA_FRAME_BYTES, decode_camera_frame


class CameraFrameProtocolTests(unittest.TestCase):
    def test_decodes_a_small_jpeg_without_a_file(self) -> None:
        image = np.full((24, 32, 3), 127, dtype=np.uint8)
        encoded, jpeg = cv2.imencode(".jpg", image)
        self.assertTrue(encoded)

        decoded = decode_camera_frame(base64.b64encode(jpeg).decode("ascii"))

        self.assertEqual(decoded.shape[:2], (24, 32))

    def test_rejects_non_image_or_oversized_payload(self) -> None:
        with self.assertRaises(ValueError):
            decode_camera_frame(base64.b64encode(b"not-an-image").decode("ascii"))
        with self.assertRaises(ValueError):
            decode_camera_frame("A" * ((MAX_CAMERA_FRAME_BYTES * 4 // 3) + 9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
