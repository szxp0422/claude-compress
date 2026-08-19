"""
calibrate_anchor.py

Run once on a sample frame to verify your anchor text and offsets
before processing a full video.

Usage:
    python calibrate_anchor.py video.mp4 "VS Code" --offset-y 28

Opens a window showing where the anchor was found (yellow box) and
the derived content ROI (green box). Press any key to close.
"""
import argparse
import json
import sys

import cv2
from detect_roi import TextAnchorTracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("anchor", help="Anchor text to search for")
    ap.add_argument("--offset-y", type=int, default=28)
    ap.add_argument("--offset-x", type=int, default=0)
    ap.add_argument("--content-height", type=int, default=None)
    ap.add_argument("--content-width", type=int, default=None)
    ap.add_argument("--search-region", nargs=4, type=int,
                    metavar=("X", "Y", "W", "H"), default=None)
    ap.add_argument("--fuzzy", type=float, default=0.7)
    ap.add_argument("--frame", type=int, default=0,
                    help="Frame index to use for calibration (default 0)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read frame {args.frame} from {args.video}")
        sys.exit(1)

    search_region = tuple(args.search_region) if args.search_region else None
    tracker = TextAnchorTracker(
        anchor_text=args.anchor,
        offset_y=args.offset_y,
        offset_x=args.offset_x,
        content_height=args.content_height,
        content_width=args.content_width,
        search_region=search_region,
        fuzzy_threshold=args.fuzzy,
    )

    result = tracker.calibrate(frame)
    print(json.dumps(result, indent=2))

    if not result["found"]:
        print("\nAnchor not found. Tips:")
        print("  - Try a shorter or more distinctive anchor string")
        print("  - Lower --fuzzy (e.g. 0.6) to accept more OCR noise")
        print("  - Try a different --frame index")
        sys.exit(1)

    # Draw boxes on a copy of the frame
    vis = frame.copy()
    ax, ay, aw, ah = result["anchor_box"]
    rx, ry, rw, rh = result["derived_roi"]

    # Yellow = anchor text location
    cv2.rectangle(vis, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 2)
    cv2.putText(vis, f"anchor: '{args.anchor}'",
                (ax, max(0, ay - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Green = derived content ROI
    cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    cv2.putText(vis, f"content ROI ({rx},{ry} {rw}x{rh})",
                (rx, max(0, ry - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    print("\nPress any key in the preview window to close.")
    cv2.imshow("Anchor calibration — yellow=anchor, green=content ROI", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nTo use these settings:")
    print(f"  python video_transcribe_roi.py {args.video} \\")
    print(f"    --anchor \"{args.anchor}\" \\")
    print(f"    --offset-y {args.offset_y} \\")
    if args.offset_x:
        print(f"    --offset-x {args.offset_x} \\")
    if args.content_height:
        print(f"    --content-height {args.content_height} \\")
    if args.content_width:
        print(f"    --content-width {args.content_width} \\")
    if search_region:
        print(f"    --search-region {' '.join(map(str, search_region))} \\")


if __name__ == "__main__":
    main()
