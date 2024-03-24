import math
import time


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects along with timestamps
        self.center_points = {}
        self.timestamps = {}
        # Keep the count of the IDs
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    self.timestamps[id] = time.time()  # Update timestamp
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                self.timestamps[self.id_count] = time.time()  # Update timestamp
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        new_timestamps = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
            new_timestamps[object_id] = self.timestamps[object_id]

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        self.timestamps = new_timestamps.copy()
        return objects_bbs_ids
