import subprocess
import time
import sys

def run_script(script_path):
    # Use subprocess to run each script in parallel
    process = subprocess.Popen([sys.executable, script_path])
    return process

def monitor_processes(processes):
    """Monitor processes and restart if any has stopped unexpectedly"""
    while True:
        time.sleep(300)  # Check every 5 minutes (or adjust as needed)
        
        for process in processes:
            # Check if the process is still running
            if process.poll() is not None:
                print(f"A script has stopped unexpectedly.")
                # Optionally, restart the script if it stops
                # Restart the process or log the status
                # Example of restarting:
                process = run_script(process.args[1])  # Restart the script
                processes.append(process)  # Add it back to the list

if __name__ == "__main__":
    # Paths to the Python scripts you want to run
    script_paths = [
        'D:/DATA/FOR_DL/objectdetection/magang2025/Pudu_Monitoring_Station/Cam1_PuduDetection_Tensor.py',
        'D:/DATA/FOR_DL/objectdetection/magang2025/Pudu_Monitoring_Station/Cam2_PuduDetection_Tensor.py',
        'D:/DATA/FOR_DL/objectdetection/magang2025/Pudu_Monitoring_Station/Cam3_PuduDetection_Tensor.py'
    ]

    # List to store process objects
    processes = []

    # Start each script in parallel
    for script in script_paths:
        process = run_script(script)
        processes.append(process)

    # Start monitoring processes
    try:
        monitor_processes(processes)
    except KeyboardInterrupt:
        print("Exiting program. Stopping all scripts.")
        for process in processes:
            process.terminate()  # Gracefully terminate each process
