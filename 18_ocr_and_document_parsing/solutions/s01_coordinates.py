"""Exercise 1: pixel box (top-left origin, 96 dpi) -> PDF points (top-left) -> PDF user space (bottom-left origin).
RUN: python s01_coordinates.py"""
PX_TO_PT = 72 / 96


def px_to_pt_topleft(box_px):
    return [round(v * PX_TO_PT, 2) for v in box_px]


def topleft_to_pdf_bottomleft(box_pt, page_h_pt):
    """[x0, y_top, x1, y_bottom_edge] with y growing DOWN  ->  [x0, y_low, x1, y_high] with y growing UP."""
    x0, y0, x1, y1 = box_pt
    return [x0, round(page_h_pt - y1, 2), x1, round(page_h_pt - y0, 2)]


box = [42, 31, 494, 64]                                   # the title block of step 07 (pixels, 800 x 640 page)
pt = px_to_pt_topleft(box)
print("points, top-left origin :", pt)
print("PDF user space (bottom-left origin, page 480 pt high):", topleft_to_pdf_bottomleft(pt, 640 * PX_TO_PT))
assert pt == [31.5, 23.25, 370.5, 48.0]
assert topleft_to_pdf_bottomleft(pt, 480.0) == [31.5, 432.0, 370.5, 456.75]
print("ok")
