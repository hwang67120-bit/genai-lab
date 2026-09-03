import unittest

import numpy as np
from PIL import Image

from genai_lab.catvton_preflight import (
    CatVTONPreflightError,
    create_catvton_input_snapshot,
)
from genai_lab.clothing import (
    CharacterAgnosticApprovedInput,
    CharacterClothingProtectionError,
    validate_character_agnostic_approved_input,
)
from genai_lab.image_digest import calculate_image_pixel_sha256


class CatVTONInputSnapshotTest(unittest.TestCase):
    def create_images(self):
        person = Image.new("RGB", (10, 10), (255, 255, 255))
        clothing = Image.new("RGB", (6, 8), (20, 40, 80))
        change_array = np.zeros((10, 10), dtype=np.uint8)
        change_array[2:8, 2:8] = 255
        protection_array = np.zeros((10, 10), dtype=np.uint8)
        protection_array[0:2, 0:2] = 255
        foreground_array = np.full((10, 10), 255, dtype=np.uint8)
        return (
            person,
            Image.fromarray(change_array, mode="L"),
            clothing,
            Image.fromarray(protection_array, mode="L"),
            Image.fromarray(foreground_array, mode="L"),
        )

    def test_snapshot_passes_with_zero_guard_overlap(self):
        images = self.create_images()
        snapshot = None
        try:
            snapshot = create_catvton_input_snapshot(*images)

            self.assertTrue(snapshot.passed)
            self.assertEqual(snapshot.change_mask_pixel_count, 36)
            self.assertEqual(snapshot.protected_overlap_pixel_count, 0)
            self.assertEqual(snapshot.outside_foreground_pixel_count, 0)
            self.assertEqual(snapshot.person_image.size, (10, 10))
            self.assertEqual(snapshot.clothing_condition_image.size, (6, 8))
            self.assertEqual(len(snapshot.person_sha256), 64)
            self.assertEqual(len(snapshot.change_mask_sha256), 64)
            self.assertEqual(len(snapshot.clothing_sha256), 64)
        finally:
            if snapshot is not None:
                snapshot.close()
            for image in images:
                image.close()

    def test_approved_model_mask_sha256_matches_preflight(self):
        model_mask = Image.new("L", (8, 8), 64)
        approved_input = CharacterAgnosticApprovedInput(
            human_agnostic_image=Image.new("RGB", (8, 8), "gray"),
            approved_change_mask=Image.new("L", (8, 8), 255),
            clothing_type="upper",
            approved_mask_pixel_count=64,
            approved_model_mask=model_mask,
            preflight_model_mask_sha256=calculate_image_pixel_sha256(
                model_mask,
                "L",
            ),
        )
        try:
            validate_character_agnostic_approved_input(
                approved_input,
                "upper",
            )
        finally:
            approved_input.close()

    def test_approved_model_mask_sha256_mismatch_is_rejected(self):
        approved_input = CharacterAgnosticApprovedInput(
            human_agnostic_image=Image.new("RGB", (8, 8), "gray"),
            approved_change_mask=Image.new("L", (8, 8), 255),
            clothing_type="upper",
            approved_mask_pixel_count=64,
            approved_model_mask=Image.new("L", (8, 8), 64),
            preflight_model_mask_sha256="0" * 64,
        )
        try:
            with self.assertRaisesRegex(
                CharacterClothingProtectionError,
                "model_mask의 SHA-256",
            ):
                validate_character_agnostic_approved_input(
                    approved_input,
                    "upper",
                )
        finally:
            approved_input.close()

    def test_snapshot_reports_protection_and_foreground_overlap(self):
        images = list(self.create_images())
        snapshot = None
        try:
            protection = np.zeros((10, 10), dtype=np.uint8)
            protection[2, 2] = 255
            images[3].close()
            images[3] = Image.fromarray(protection, mode="L")

            foreground = np.full((10, 10), 255, dtype=np.uint8)
            foreground[7, 7] = 0
            images[4].close()
            images[4] = Image.fromarray(foreground, mode="L")

            snapshot = create_catvton_input_snapshot(*images)

            self.assertFalse(snapshot.passed)
            self.assertEqual(snapshot.protected_overlap_pixel_count, 1)
            self.assertEqual(snapshot.outside_foreground_pixel_count, 1)
        finally:
            if snapshot is not None:
                snapshot.close()
            for image in images:
                image.close()

    def test_snapshot_rejects_coordinate_mismatch(self):
        images = list(self.create_images())
        try:
            images[4].close()
            images[4] = Image.new("L", (9, 10), 255)

            with self.assertRaisesRegex(
                CatVTONPreflightError,
                "캐릭터 외곽 좌표 불일치",
            ):
                create_catvton_input_snapshot(*images)
        finally:
            for image in images:
                image.close()


if __name__ == "__main__":
    unittest.main()
