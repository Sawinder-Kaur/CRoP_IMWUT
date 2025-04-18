README: Recreating Evaluation on Google Pixel 6 using Aidlux Framework
================================================

Step 1: Get Aidlux Platform APK
-------------------------------
Download the latest APK from: https://aidlux.com/platform

Step 2: Enable Developer Options on Your Phone
----------------------------------------------
1. Go to Settings > About phone.
2. Tap 'Build number' seven times until you see a message saying you're a developer.
3. Go to Settings > System > Developer options.
4. Enable 'USB debugging'.

Step 3: Connect Your Phone
--------------------------
1. Connect your phone to your computer using a USB cable.
2. When prompted, select “File Transfer” or “MTP” mode.

Step 4: Verify Device is Detected
---------------------------------
Open a terminal or command prompt on your computer and run:
    adb devices
You should see your device listed. If it asks for permission on your phone, allow it.

Step 5: Install the Aidlux APK
------------------------------
Run the following command in terminal:
    adb install /path/to/your/app.apk
Replace '/path/to/your/app.apk' with the actual path to the downloaded Aidlux APK.
If the path contains spaces, wrap it in quotes.

Step 6: Set Up the Python Environment
-------------------------------------
1. Open the Aidlux app on your phone.
2. In the integrated terminal, recreate the environment using the provided `requirements.txt` file.

Step 7: Move Python Files to Phone
----------------------------------
1. Use Android Studio or any preferred method to move the `.py` files to the phone.

Step 8: Run Your Script
-----------------------
1. Open the integrated terminal in Aidlux.
2. Run the script using:
    python3 filename.py
Replace `filename.py` with the actual script name.

================================================
End of Instructions
