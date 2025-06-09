import cv2
from datetime import datetime, timedelta
from ultralytics import YOLO
import numpy as np
import mysql.connector
import traceback
from urllib.parse import quote
import queue
import threading
import time
import torch
import os
import signal
import logging
import sys
from mysql.connector import pooling
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

# Configure logging with more detailed output for debugging
logging.basicConfig(
    level=logging.DEBUG,  # Adding DEBUG for more detailed logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("robot_tracker.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RobotTracker")

# Flag to control the running state of the application
running = True
reconnect_delay = 5  # seconds to wait before reconnecting
max_queue_size = 100  # Maximum size for the queue to hold frames
cleanup_interval = 3600  # Clean old data every 1 hour (in seconds)
max_record_age = 86400  # Store data for a maximum of 24 hours (in seconds)
sec_stop = 3  # Stop detection after 3 seconds of inactivity

# Queue to store frames with a size limit
frame_queue = queue.Queue(maxsize=max_queue_size)

class MySQLDatabasePool:
    def __init__(self, config):
        self.config = config
        self.pool = None
        self.create_pool()
        
    def create_pool(self):
        """Create the MySQL connection pool"""
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name="robot_tracker_pool",
                pool_size=5,
                host=self.config['host'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                autocommit=True,
                connect_timeout=10
            )
            logger.info("Database connection pool created successfully")
        except mysql.connector.Error as err:
            logger.error(f"Error creating connection pool: {err}")
            self.pool = None
            # Schedule reconnection
            threading.Timer(reconnect_delay, self.create_pool).start()
    
    def get_connection(self):
        """Get a connection from the pool with error handling"""
        if not self.pool:
            self.create_pool()
            if not self.pool:
                return None
                
        try:
            return self.pool.get_connection()
        except mysql.connector.Error as err:
            logger.error(f"Failed to get connection from pool: {err}")
            # Recreate pool on failure
            self.pool = None
            self.create_pool()
            return None
    
    def insert(self, data):
        """Insert data into the database"""
        conn = self.get_connection()
        if not conn:
            logger.error("Failed to insert data - no connection available")
            return None
            
        try:
            cursor = conn.cursor()
            query = f"""
                INSERT INTO {self.config['table']} 
                (location, station, status, timestamp)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, data)
            conn.commit()
            last_id = cursor.lastrowid
            cursor.close()
            return last_id
        except mysql.connector.Error as err:
            logger.error(f"Database insert error: {err}")
            return None
        finally:
            conn.close()

    def close(self):
        """Close the connection pool"""
        if self.pool:
            logger.info("Closing database connection pool")
            self.pool = None

class CheckpointTracker:
    def __init__(self, model_path, checkpoint_areas, db_handler=None, location="CAM1"):
        """Initialize YOLO model and tracker"""
        try:
            self.model = YOLO(model_path)
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                logger.info("Using CUDA for inference")
            else:
                self.model = self.model.to('cpu')
                logger.info("Using CPU for inference")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        self.checkpoint_areas = checkpoint_areas
        self.robot_records = {}
        self.current_in_checkpoint = {area: set() for area in checkpoint_areas.keys()}
        self.pending_exits = {}
        self.stop_detection = {}
        self.stop_already_triggered = set()
        self.active_events = []
        self.db = db_handler
        self.location = location
        
        self.robot_id = None
        self.robot_last_entry_area = {}
        self.robot_db_entries = {}
        
        self.last_cleanup_time = time.time()
        
        self.frames_processed = 0
        self.detection_times = []
        
        logger.info(f"Checkpoint Tracker initialized with {len(checkpoint_areas)} areas")

        # New addition
        self.entry_time_limit = 3  # Seconds to wait for DB IN after an entry is detected
        self.robot_in_pending_db = {}  # To track robot entries waiting for DB IN


    def is_in_area(self, x, y, area_coords):
        """Check if the robot is inside a specific area"""
        x1, y1, x2, y2 = area_coords
        return x1 <= x <= x2 and y1 <= y <= y2

    def process_frame(self, frame, conf_threshold=0.6):
        """Process a frame with object detection"""
        try:
            start_time = time.time()
            results = self.model.track(frame, persist=True, verbose=False, conf=conf_threshold)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().numpy()

                largest_box = None
                largest_area = 0
                largest_track_id = None

                for i, (x, y, w, h) in enumerate(boxes):
                    track_id = int(track_ids[i])
                    class_id = int(classes[i])

                    if class_id != 0:  # Skip if not a Pudu robot
                        continue

                    area = w * h
                    if area > largest_area:
                        largest_area = area
                        largest_box = (x, y, w, h)
                        largest_track_id = track_id

                if largest_box:
                    if self.robot_id is None:  
                        self.robot_id = largest_track_id  
                        logger.info(f"First Pudu robot detected with ID: {self.robot_id}")

                    track_id = self.robot_id
                    x, y, w, h = largest_box
                    center_x, center_y = int(x), int(y)

                    robot_in_any_area = False
                    for area_name, coords in self.checkpoint_areas.items():
                        key = (track_id, area_name)

                        if self.is_in_area(center_x, center_y, coords):
                            robot_in_any_area = True
                            
                            if key in self.pending_exits:
                                del self.pending_exits[key]
                                self.stop_detection[key] = datetime.now()

                            if track_id not in self.current_in_checkpoint[area_name]:
                                if key not in self.stop_detection or (datetime.now() - self.stop_detection[key]).total_seconds() > sec_stop:
                                    self._handle_entry(track_id, area_name)

                            if key in self.stop_detection and key not in self.stop_already_triggered:
                                if (datetime.now() - self.stop_detection[key]).total_seconds() > sec_stop:
                                    logger.info(f"Robot {track_id} STOPPED at {area_name} (Entered: {self.stop_detection[key].strftime('%Y-%m-%d %H:%M:%S')})")
                                    self.stop_already_triggered.add(key)
                                    
                                    if self.db:
                                        try:
                                            entry_id = self.db.insert((self.location, area_name, 'In', datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                                            if entry_id:
                                                self.robot_db_entries[key] = {'entry_id': entry_id, 'area': area_name, 'timestamp': datetime.now()}
                                                logger.info(f"Robot {track_id} [DB IN] {area_name} (ID: {entry_id})")
                                        except Exception as e:
                                            logger.error(f"Database insert error (IN): {e}")

                    for area_name in self.checkpoint_areas.keys():
                        key = (track_id, area_name)
                        if not self.is_in_area(center_x, center_y, self.checkpoint_areas[area_name]):
                            if track_id in self.current_in_checkpoint[area_name] and key not in self.pending_exits:
                                self.pending_exits[key] = datetime.now()

                    self._draw_robot_info(frame, x, y, w, h, track_id)

            self._process_pending_exits()
            self._draw_checkpoint_areas(frame)
            self._draw_events(frame)

            self.frames_processed += 1
            proc_time = time.time() - start_time
            self.detection_times.append(proc_time)

            if time.time() - self.last_cleanup_time > cleanup_interval:
                self._cleanup_old_data()
                self.last_cleanup_time = time.time()
                
                if len(self.detection_times) > 0:
                    avg_time = sum(self.detection_times) / len(self.detection_times)
                    logger.info(f"Performance stats: {self.frames_processed} frames processed, " 
                                f"avg processing time: {avg_time:.4f}s per frame")
                    self.detection_times = []  # Reset
                
            return frame
            
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            logger.error(traceback.format_exc())
            return frame  # Return original frame on error
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory leaks"""
        try:
            now = time.time()
            current_time = datetime.now()
            logger.info("Performing data cleanup")
            
            # Clean robot_records that are too old
            for robot_id in list(self.robot_records.keys()):
                for area_name in list(self.robot_records[robot_id].keys()):
                    # Filter records that are still recent
                    fresh_records = []
                    for record in self.robot_records[robot_id][area_name]:
                        # If record has exit and is less than max_record_age, keep it
                        if record['exit'] is not None:
                            record_age = (current_time - record['exit']).total_seconds()
                            if record_age < max_record_age:
                                fresh_records.append(record)
                        else:
                            # Records without exit (still active) are always kept
                            fresh_records.append(record)
                    
                    if fresh_records:
                        self.robot_records[robot_id][area_name] = fresh_records
                    else:
                        # Remove area if no records left
                        del self.robot_records[robot_id][area_name]
                
                # Remove robot if no area is left
                if not self.robot_records[robot_id]:
                    del self.robot_records[robot_id]
            
            # Clean active_events that are too old
            self.active_events = [event for event in self.active_events 
                                  if (current_time - event['timestamp']).total_seconds() < 5]
                                  
            # Clean pending_exits that are too old
            for key in list(self.pending_exits.keys()):
                if (current_time - self.pending_exits[key]).total_seconds() > 30:  # 1 minute
                    del self.pending_exits[key]
                    
            # Clean stop_detection that are too old
            for key in list(self.stop_detection.keys()):
                if (current_time - self.stop_detection[key]).total_seconds() > 60:  # 5 minutes
                    del self.stop_detection[key]
                    
            # Clean robot_db_entries that are too old
            for key in list(self.robot_db_entries.keys()):
                if (current_time - self.robot_db_entries[key]['timestamp']).total_seconds() > 3600:  # 1 hour
                    del self.robot_db_entries[key]
                    
            logger.info(f"Data cleanup completed. Records for {len(self.robot_records)} robots remain.")
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")

    def _process_pending_exits(self):
        """Process pending exits for robots"""
        now = datetime.now()
        to_remove = []
        
        for (track_id, area_name), exit_time in self.pending_exits.items():
            if (now - exit_time).total_seconds() > 0:  
                if track_id in self.robot_last_entry_area and self.robot_last_entry_area[track_id] == area_name:
                    self._handle_exit(track_id, area_name, exit_time)
                to_remove.append((track_id, area_name))
        
        for key in to_remove:
            del self.pending_exits[key]

    def _handle_entry(self, track_id, area_name):
        """Handle entry detection for the robot"""
        current_time = datetime.now()
        key = (track_id, area_name)
        
        # Check if entry was previously flagged as needing DB IN
        if key in self.robot_in_pending_db:
            elapsed_time = (current_time - self.robot_in_pending_db[key]).total_seconds()
            if elapsed_time > self.entry_time_limit:
                logger.info(f"Robot {track_id} [IN] {area_name} not inserted to DB in time, deleting")
                # Remove the record if it wasn't inserted into DB within the time limit
                del self.robot_records[track_id][area_name]
                self.current_in_checkpoint[area_name].discard(track_id)
                del self.robot_in_pending_db[key]
                return

        if key in self.stop_detection:
            del self.stop_detection[key]
        self.stop_already_triggered.discard(key)

        if track_id not in self.robot_records:
            self.robot_records[track_id] = {}
            
        if area_name not in self.robot_records[track_id]:
            self.robot_records[track_id][area_name] = []
            
        logger.info(f"Robot {track_id} [IN] {area_name}")
        self.robot_records[track_id][area_name].append({
            'entry': current_time,
            'exit': None,
            'status': 'In',
            'db_in_sent': False
        })
        self.current_in_checkpoint[area_name].add(track_id)
        self.stop_detection[key] = current_time
        self.stop_already_triggered.discard(key)
        
        self.robot_last_entry_area[track_id] = area_name

        # Add to the pending DB IN list
        self.robot_in_pending_db[key] = current_time
    
    def _handle_db_in(self, track_id, area_name):
        """Called when DB IN is successfully added"""
        key = (track_id, area_name)
        if key in self.robot_in_pending_db:
            # DB IN was received, remove from the pending list
            del self.robot_in_pending_db[key]
            logger.info(f"Robot {track_id} [DB IN] {area_name} successfully inserted")
            # Set db_in_sent to True
            for record in self.robot_records[track_id][area_name]:
                if record['status'] == 'In' and not record['db_in_sent']:
                    record['db_in_sent'] = True
                    break

    def _handle_exit(self, track_id, area_name, exit_time):
        """Handle exit detection for the robot"""
        key = (track_id, area_name)
        
        if key in self.stop_detection:
            elapsed_time = (exit_time - self.stop_detection[key]).total_seconds()
            if elapsed_time < 1:  # If the robot hasn't stopped long enough, ignore the exit
                logger.debug(f"Robot {track_id} incorrect detection at {area_name}, not exiting")
                return

        if track_id in self.robot_last_entry_area and self.robot_last_entry_area[track_id] != area_name:
            logger.warning(f"Robot {track_id} OUT ignored: exit area {area_name} doesn't match last entry {self.robot_last_entry_area[track_id]}")
            return

        logger.info(f"Robot {track_id} [OUT] {area_name}")
        self.current_in_checkpoint[area_name].discard(track_id)
        
        if track_id in self.robot_records and area_name in self.robot_records[track_id]:
            for record in reversed(self.robot_records[track_id][area_name]):
                if record['exit'] is None:
                    record['exit'] = exit_time
                    record['status'] = 'Out'
                    break
        
        if key in self.stop_detection:
            del self.stop_detection[key]
        self.stop_already_triggered.discard(key)

        if self.db and key in self.robot_db_entries:
            try:
                self.db.insert((self.location, area_name, 'Out', exit_time.strftime("%Y-%m-%d %H:%M:%S")))
                logger.info(f"Robot {track_id} [DB OUT] {area_name}")
                del self.robot_db_entries[key]
            except Exception as e:
                logger.error(f"Failed to insert to database (OUT): {e}")

        area_coords = self.checkpoint_areas[area_name]
        self.active_events.append({
            'type': 'exit',
            'message': f"EXITED {area_name}",
            'position': (area_coords[0] + 10, area_coords[1] + 60),
            'timestamp': exit_time
        })

    def _draw_events(self, frame):
        """Draw active events on the frame"""
        now = datetime.now()
        to_remove = []
        
        for idx, event in enumerate(self.active_events):
            elapsed = (now - event['timestamp']).total_seconds()
            
            if elapsed > 5:  # Show message for 5 seconds
                to_remove.append(idx)
                continue
                
            color = (0, 255, 0) if event['type'] == 'entry' else (0, 0, 255)
            text_size, _ = cv2.getTextSize(event['message'], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x, y = event['position']
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y - text_size[1] - 5), (x + text_size[0], y + 5), color, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, event['message'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        for idx in reversed(to_remove):
            del self.active_events[idx]

    def _draw_robot_info(self, frame, x, y, w, h, track_id):
        """Draw robot information on the frame"""
        center_x, center_y = int(x), int(y)
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        in_checkpoint = any(track_id in area for area in self.current_in_checkpoint.values())
        color = (0, 255, 255) if in_checkpoint else (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        display_text = f"ID: {track_id}"
        if track_id in self.robot_last_entry_area:
            display_text += f" | Area: {self.robot_last_entry_area[track_id]}"

        cv2.putText(frame, display_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    def _draw_checkpoint_areas(self, frame):
        """Draw checkpoint areas on the frame"""
        for area_name, (x1, y1, x2, y2) in self.checkpoint_areas.items():
            station_color = (31, 95, 255)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), station_color, -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            text_color = (31, 95, 255)
            cv2.putText(frame, area_name, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

def setup_signal_handlers():
    def signal_handler(sig, frame):
        global running
        logger.info("Shutdown signal received")
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def Receive(url, frame_queue):
    global running, frame_counter
    frame_counter = 0
    skip_frames = 2
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    backoff_factor = 1.5
    
    logger.info(f"Starting video capture from {url}")
    
    while running:
        try:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video source: {url}")
                
            logger.info("Video source connected successfully")
            reconnect_attempts = 0  # Reset on successful connection
            
            while running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to receive frame, reconnecting...")
                    break
                
                frame_counter += 1
                if frame_counter % skip_frames == 0:
                    frame = cv2.resize(frame, (640, 480))
                    try:
                        frame_queue.put(frame, block=False)
                    except queue.Full:
                        pass
            cap.release()
            logger.info("Video connection lost, attempting to reconnect")
            
        except Exception as e:
            logger.error(f"Video capture error: {e}")
            if cap:
                cap.release()
        
        reconnect_attempts += 1
        if reconnect_attempts > max_reconnect_attempts:
            delay = min(30, reconnect_delay * (backoff_factor ** (reconnect_attempts - 1)))
        else:
            delay = reconnect_delay
        
        logger.info(f"Waiting {delay}s before reconnection attempt {reconnect_attempts}")
        
        if not running:
            break
            
        time.sleep(delay)
    
    logger.info("Video capture thread stopping")

def Display(tracker, frame_queue):
    global running
    last_frame_time = time.time()
    fps_counter = 0
    fps = 0
    
    logger.info("Starting display thread")
    
    while running:
        try:
            try:
                frame = frame_queue.get(timeout=1)
                fps_counter += 1
                
                current_time = time.time()
                if current_time - last_frame_time >= 1.0:
                    fps = fps_counter / (current_time - last_frame_time)
                    fps_counter = 0
                    last_frame_time = current_time
                
                processed_frame = tracker.process_frame(frame)
                
                text = f"FPS: {fps:.1f}"
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                overlay = processed_frame.copy()
                cv2.rectangle(overlay, (5, 10), (15 + text_width, 40), (0, 0, 0), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, processed_frame, 1 - alpha, 0, processed_frame)
                cv2.putText(processed_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if tracker.robot_id:
                    count_areas = sum(1 for areas in tracker.current_in_checkpoint.values() 
                                    if tracker.robot_id in areas)
                    text = f"Robot in {count_areas} areas"
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    overlay = processed_frame.copy()
                    cv2.rectangle(overlay, (5, 50), (15 + text_width, 80), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, alpha, processed_frame, 1 - alpha, 0, processed_frame)
                    cv2.putText(processed_frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Robot Tracking", processed_frame)
                
            except queue.Empty:
                pass
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User requested exit (q key)")
                running = False
                break
                
        except Exception as e:
            logger.error(f"Display error: {e}")
            logger.error(traceback.format_exc())
            time.sleep(1)
    
    logger.info("Display thread stopping")
    cv2.destroyAllWindows()

# Configure Checkpoint Areas for 640x480 frame (Adjusted)
CHECKPOINT_AREAS = {
    'Station 0': (int(771 * (640/1920)), int(20 * (480/1080)), int(854 * (640/1920)), int(115 * (480/1080))),
    'Station 1': (int(833 * (640/1920)), int(135 * (480/1080)), int(934 * (640/1920)), int(251 * (480/1080))),
    'Station 9': (int(701 * (640/1920)), int(155 * (480/1080)), int(804 * (640/1920)), int(255 * (480/1080))),
    'Station 2': (int(597 * (640/1920)), int(654 * (480/1080)), int(807 * (640/1920)), int(911 * (480/1080))),
    'Station 8': (int(393 * (640/1920)), int(770 * (480/1080)), int(580 * (640/1920)), int(1038 * (480/1080)))
}

def main():
    global running
    setup_signal_handlers()
    
    try:
        logger.info("=== Starting Pudu Robot Tracking System ===")
        
        db_config = {
            'host': 'ip_database',
            'user': 'username',
            'password': 'password',
            'database': 'database_name',
            'table': 'table_name'
        }
        
        model_path = 'path_to_model.pt'
        # model_path = 'path_to_model.engine'
        if not os.path.exists(model_path):
            logger.error(f"Model not found at path: {model_path}")
            return

        logger.info("Initializing database connection pool")
        db_pool = MySQLDatabasePool(db_config)
        
        logger.info("Initializing robot tracker")
        tracker = CheckpointTracker(model_path, CHECKPOINT_AREAS, db_handler=db_pool, location="your_location") # Change with your location

        username = "username"
        password = "password"
        ip = "your_ip_cctv"
        port = "port"
        
        use_rtsp = True
        
        if use_rtsp:
            url = f"rtsp://{username}:{quote(password)}@{ip}:{port}"
            logger.info(f"Using RTSP stream: {ip}:{port}")
        else:
            url = "Path_to_video.mp4"
            logger.info(f"Using video file: {url}")

        threads = []
        receive_thread = threading.Thread(target=Receive, args=(url, frame_queue), name="ReceiveThread")
        display_thread = threading.Thread(target=Display, args=(tracker, frame_queue), name="DisplayThread")
        
        receive_thread.daemon = True
        display_thread.daemon = True
        
        threads.append(receive_thread)
        threads.append(display_thread)
        
        receive_thread.start()
        display_thread.start()
        
        logger.info("All threads started")
        
        while running:
            for i, thread in enumerate(threads):
                if not thread.is_alive():
                    logger.warning(f"Thread '{thread.name}' died, restarting")
                    
                    if thread.name == "ReceiveThread":
                        new_thread = threading.Thread(target=Receive, args=(url, frame_queue), name="ReceiveThread")
                    else:
                        new_thread = threading.Thread(target=Display, args=(tracker, frame_queue), name="DisplayThread")
                    
                    new_thread.daemon = True
                    new_thread.start()
                    threads[i] = new_thread
            
            time.sleep(5)
        
        logger.info("Main program exiting, waiting for threads to finish")
        for thread in threads:
            thread.join(timeout=3.0)
            
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        logger.error(traceback.format_exc())
    finally:
        running = False
        logger.info("Cleaning up resources")
        cv2.destroyAllWindows()
        
        if 'tracker' in locals():
            logger.info("\nROBOT VISIT HISTORY:")
            for robot_id, checkpoints in tracker.robot_records.items():
                logger.info(f"\nRobot {robot_id}:")
                for area, records in checkpoints.items():
                    for rec in records:
                        exit_time = rec['exit'].strftime("%Y-%m-%d %H:%M:%S") if rec['exit'] else "Still in area"
                        logger.info(f" - {area}: {rec['entry'].strftime('%Y-%m-%d %H:%M:%S')} - {exit_time}")

        if 'db_pool' in locals() and db_pool:
            db_pool.close()
            
        logger.info("=== Pudu Robot Tracking System Stopped ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
