import cv2, numpy as np
f = np.load("captures/session_2026-09-29_17-17-12/frames.npy", mmap_mode="r")
i, play = 0, False
while True:
    cv2.imshow("frames", f[i]); cv2.setWindowTitle("frames", f"frame {i}/{len(f)-1}")
    k = cv2.waitKey(10 if play else 0) & 0xFF
    if k == ord("q"): break
    if k == ord(" "): play = not play
    elif k in (81, ord("a")): i = max(i - 1, 0)          # left
    elif k in (83, ord("d")) or play: i = min(i + 1, len(f) - 1)  # right
cv2.destroyAllWindows()
16-46-41