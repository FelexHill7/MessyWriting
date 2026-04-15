"""
main.py — Single entry point for the RxRead pipeline.

Delegates to services/ and web/ packages so nothing else needs to be
run directly.  All paths and configuration live in config.py.

Usage:
    python main.py                    Train the model, then launch the web UI
    python main.py train              Train the CRNN, then launch the web UI
    python main.py serve              Launch the web UI only (skip training)
    python main.py predict <image>    Quick CLI inference on a single image
"""

import os
import sys

# Ensure project root is the working directory regardless of where the script
# is invoked from — keeps data/ and weight paths consistent.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


def main():
    command = sys.argv[1].lower() if len(sys.argv) >= 2 else "all"

    if command == "train":
        from services.training import train
        train()
        print("\nTraining complete. Launching web UI...\n")
        from web.app import run_server
        run_server()

    elif command == "serve":
        from web.app import run_server
        run_server()

    elif command == "predict":
        if len(sys.argv) < 3:
            print("Usage: python main.py predict <image_path>")
            return
        from services.inference import predict_file
        print(predict_file(sys.argv[2]))

    elif command == "all":
        # Full pipeline: train (if no weights exist) → launch web UI
        from config import BEST_WEIGHTS, FINAL_WEIGHTS
        weights_exist = (
            os.path.exists(BEST_WEIGHTS)
            or os.path.exists(FINAL_WEIGHTS)
        )

        if not weights_exist:
            print("No trained model found — starting training...\n")
            from services.training import train
            train()
            print()
        else:
            print("Trained model found — skipping training.")

        print("Launching web UI...\n")
        from web.app import run_server
        run_server()

    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [train | serve | predict <image>]")


if __name__ == "__main__":
    main()
