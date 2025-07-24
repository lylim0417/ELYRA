
# NFS Setup Instructions for ELYRA System

This section explains how to set up **NFS (Network File System)** for file sharing between the Raspberry Pi (client) and the server.

## NFS Server Setup (on the server machine)

1. **Install NFS Server**

   Install the NFS server package on your server machine:
   ```bash
   sudo apt install nfs-kernel-server
   ```

2. **Enable NFS at Boot**

   Enable the NFS server to start automatically at boot:
   ```bash
   sudo systemctl enable --now nfs-server
   ```

3. **Create Directory for Sharing**

   Create a directory to be shared via NFS:
   ```bash
   sudo mkdir -p /mnt/nfs/recordings
   ```

4. **Configure NFS Exports**

   Open the `/etc/exports` file to configure the NFS share:
   ```bash
   sudo nano /etc/exports
   ```

   Add the following line to specify the directory to share and who can access it:
   ```
   /mnt/nfs/recordings 192.168.1.15(rw,async,all_squash,insecure,no_subtree_check)
   ```
   **Explanation of options**:
   - `rw`: Allows read-write access.
   - `async`: Enables asynchronous writes for better performance.
   - `all_squash`: Maps all requests to the `nobody` user for security.
   - `insecure`: Allows connections from non-privileged ports (useful for Raspberry Pi).
   - `no_subtree_check`: Disables subtree checking for better performance.

5. **Apply NFS Configuration**

   After saving the `/etc/exports` file, apply the changes with:
   ```bash
   sudo exportfs -arv
   ```

---

## NFS Client Setup (on Raspberry Pi)

1. **Install NFS Client**

   Install the NFS client package on the Raspberry Pi:
   ```bash
   sudo apt install nfs-common
   ```

2. **Mount NFS Share**

   Mount the shared NFS directory from the server to the Raspberry Pi:
   ```bash
   sudo mount -t nfs4 192.168.1.20:/mnt/nfs/recordings /mnt/nfs/recordings
   ```

   Replace `192.168.1.20` with the IP address of your NFS server.

3. **Automatically Mount NFS Share at Boot**

   To ensure the NFS share is mounted automatically at boot, add it to the `/etc/fstab` file on the Raspberry Pi:
   ```bash
   sudo nano /etc/fstab
   ```

   Add the following line:
   ```
   192.168.1.20:/mnt/nfs/recordings /mnt/nfs_recordings nfs4 rw,nosuid,nodev,noexec,_netdev 0 0
   ```

4. **Verify NFS Mount**

   After mounting the NFS share, verify it by checking the available disk space:
   ```bash
   df -h
   ```

   The output should show the NFS mount, similar to:
   ```
   192.168.1.20:/mnt/nfs/recordings   25G  15G   8.9G  63% /mnt/nfs/recordings
   ```

---

## Automating File Management with Python

To automate the process of managing video files, you will write a Python script that will:
- Check if today's video directory exists.
- List the video files in that directory.
- Compare the files with the contents of the NFS share to see which files haven't been uploaded yet.
- Upload only the files that are new.

### 1. **Write Python Script to Automate File Upload**

Create a Python script `RaspiMountToServer.py` to automate file management between the Raspberry Pi and the NFS server.

### 2. **Create a Log File**

Create a log file to capture the output or errors of the Python script:
```bash
touch /home/lylim/AIoTCam/nfs_mount_logfile.log
sudo chmod 755 /home/lylim/AIoTCam/nfs_mount_logfile.log
```

### 3. **Set Up a Cron Job to Run the Script Every 5 Minutes**

Automate the Python script to run every 5 minutes using `cron`:
```bash
crontab -e
```

Add the following line to schedule the script:
```bash
*/5 * * * * /usr/bin/python3 /home/lylim/AIoTCam/RaspiMountToServer.py >> /home/lylim/AIoTCam/nfs_mount_logfile.log 2>&1
```

### 4. **Configure Log Rotation**

To manage log files, set up log rotation. Create a configuration file for **logrotate**:
```bash
sudo nano /etc/logrotate.d/nfs_mount_logfile
```

Add the following configuration to rotate logs daily and keep 7 days of logs:
```
/home/lylim/AIoTCam/nfs_mount_logfile.log {
    daily
    rotate 7
    compress
    notifempty
    create 0640 lylim lylim
    sharedscripts
    postrotate
        systemctl reload rsyslog >/dev/null 2>&1 || true
    endscript
}
```

### 5. **Test Logrotate Configuration**

Test the log rotation configuration to ensure it is working correctly:
```bash
sudo logrotate -d /etc/logrotate.d/nfs_mount_logfile
```

---

## **Conclusion**

By following these instructions, you will have a working **NFS server and client** setup for sharing files between your Raspberry Pi and the server. The Python script will automate the process of checking and uploading new video files to the NFS share, and the cron job will schedule the script to run every 5 minutes.

This setup will ensure efficient file management and real-time monitoring of PPE compliance data.
