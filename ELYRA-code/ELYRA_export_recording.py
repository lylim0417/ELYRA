import os
import subprocess
from datetime import datetime, timedelta

# Constants
OUTPUT_BASE_DIR = "/home/ELYRA-code/Output"
TODAY_DIR_NAME = datetime.now().strftime('%Y-%m-%d')
TODAY_DIR_PATH = os.path.join(OUTPUT_BASE_DIR, TODAY_DIR_NAME)
NFS_MOUNT_POINT = "/mnt/nfs/recordings"
NFS_DIRECTORY = os.path.join(NFS_MOUNT_POINT, TODAY_DIR_NAME)

print("******************************************************************************")
print(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# Function to check and mount NFS share
def mount_nfs():
    # Check if NFS is already mounted
    result = subprocess.run(['mount'], stdout=subprocess.PIPE, text=True)
    if NFS_MOUNT_POINT not in result.stdout:
        # Mount the NFS share if not already mounted
        subprocess.run(['sudo', 'mount', NFS_MOUNT_POINT])
        print(f"NFS 1st time mount: {NFS_MOUNT_POINT}")
    else:
        print(f"NFS already mount: {NFS_MOUNT_POINT}")


# Function to copy videos that haven't been uploaded
def copy_new_videos():
    if os.path.exists(TODAY_DIR_PATH):
        # Get a list of video files in the local directory
        local_videos = os.listdir(TODAY_DIR_PATH)

        # Create today's directory is it doesn't exist
        os.makedirs(NFS_DIRECTORY, exist_ok=True)

        # Check for already uploaded videos in the NFS mount
        try:
            nfs_videos = subprocess.check_output(['ls', NFS_DIRECTORY], text=True).splitlines()
        except subprocess.CalledProcessError as e:
            print(f"Error accessing NFS share: {e}")
            return

        # Upload files that have not been uploaded yet
        for video in local_videos:
            if not(video.startswith('temp_')) and video not in nfs_videos:  # If the video is not in NFS
                today_video_path = os.path.join(TODAY_DIR_PATH, video)
                try:
                    subprocess.run(['cp', today_video_path, NFS_DIRECTORY], check=True)
                    print(f"Uploaded: {video}")
                except subprocess.CalledProcessError as e:
                    print(f"Error uploading {video}: {e}")


# Delete previous's directory
def delete_non_today_directories():
    # List all directories in OUTPUT_BASE_DIR
    for item in os.listdir(OUTPUT_BASE_DIR):
        item_path = os.path.join(OUTPUT_BASE_DIR, item)
        # Check if it's a directory and NOT today's directory
        if os.path.isdir(item_path) and item != TODAY_DIR_NAME:
            # Delete the directory and all its contents
            os.system(f"rm -rf {item_path}")
            print(f"Deleted: {item_path}")


# Main execution
if __name__ == "__main__":
    mount_nfs()
    copy_new_videos()  # Only copy new videos
    delete_non_today_directories()
