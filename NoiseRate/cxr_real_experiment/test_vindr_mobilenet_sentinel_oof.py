import unittest

import torch

from vindr_mobilenet_sentinel_oof import build_parser, create_mobilenet


class MobileNetSentinelOOFTest(unittest.TestCase):
    def test_model_is_one_channel_six_output_mobilenet(self) -> None:
        model = create_mobilenet()
        self.assertEqual(model.features[0][0].in_channels, 1)
        self.assertEqual(model.classifier[-1].out_features, 6)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(1, 1, 224, 224)).shape), (1, 6))

    def test_training_defaults_match_locked_full_pool_design(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--blind-cohort", "action.csv",
                "--sentinel-cohort", "sentinel.csv",
                "--split-reference-cohort", "split.csv",
                "--image-index", "index.csv",
                "--image-root", "images",
                "--output-dir", "output",
                "--seed", "11003",
            ]
        )
        self.assertEqual(args.n_splits, 4)
        self.assertEqual(args.epochs, 50)
        self.assertEqual(args.early_stopping_patience, 8)
        self.assertEqual(args.learning_rate, 1e-3)
        self.assertEqual(args.batch_size, 32)


if __name__ == "__main__":
    unittest.main()
