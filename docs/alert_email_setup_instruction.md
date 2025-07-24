
# Alert Email Setup Instructions for ELYRA System

This section explains how to set up the **email alert system** using **SMTP** and **MIME** to notify users when PPE violations are detected. The **alert email script** runs on the **server** side to send out notifications when non-compliance is identified.

## Setting Up the Cron Job for Email Alerts (on **server**)

1. **Ensure the Email Alert Script is in Place**

   First, make sure you have the email alert script [ELYRA_alert_email.py](../ELYRA-code/ELYRA_alert_email.py) ready and placed on your **server** machine. This script will be responsible for sending email notifications when PPE violations are detected.

2. **Install Dependencies**

   Install necessary libraries and packages:
   ```bash
   pip install -r requirements/server_requirements.txt
   ```
   
2. **Create a Log File for Cron Job**

   Create a log file where the output or errors of the script can be captured:
   ```bash
   touch /home/ELYRA-code/email_alert_logfile.log
   sudo chmod 755 /home/ELYRA-code/email_alert_logfile.log
   ```

3. **Set Up a Cron Job to Run the Script on **Server****

   Set up a cron job to run the **`ELYRA_alert_email.py`** script at regular intervals (e.g., every 5 minutes) on the **server** to check for PPE violations and send the email alerts.

   - Open the **crontab** configuration file:
     ```bash
     crontab -e
     ```

   - Add the following line to schedule the script. This will run the script every 5 minutes:
     ```bash
     */5 * * * * /usr/bin/python3 /home/ELYRA-code/ELYRA_alert_email.py >> /home/ELYRA-code/email_alert_logfile.log 2>&1
     ```

   **Explanation**:
   - `*/5 * * * *` tells cron to run the script every 5 minutes.
   - `/usr/bin/python3` is the Python interpreter. Make sure to replace it with the correct path if needed.
   - `/home/lylim/AIoTCam/ELYRA_alert_email.py` is the path to the alert email script on the **server**.
   - The output and errors will be logged to **`email_alert_logfile.log`**.

4. **Verify the Cron Job**

   To confirm that the cron job is set up correctly, you can list your current cron jobs:
   ```bash
   crontab -l
   ```

   You should see the following line in the list:
   ```bash
   */5 * * * * /usr/bin/python3 /home/ELYRA-code/ELYRA_alert_email.py >> /home/ELYRA-code/email_alert_logfile.log 2>&1
   ```

5. **Check the Log File**

   After the cron job runs, check the log file for any output or errors:
   ```bash
   cat /home/lylim/AIoTCam/email_alert_logfile.log
   ```

   This will help you confirm that the script is running as expected and sending email alerts.

---

## **Conclusion**

By following these instructions, you will have successfully set up a cron job to automatically run the **`ELYRA_alert_email.py`** script on the **server**. This script will send email alerts when PPE violations are detected, ensuring real-time monitoring and notifications.
