const { app, BrowserWindow, session, ipcMain, shell } = require('electron');
const path = require('path');
const { exec, spawn } = require('child_process');

// Global reference to the ffmpeg process so we can manage it.
let ffmpegProcess = null;

/**
 * Creates the main application window.
 */
function createWindow() {
    const mainWindow = new BrowserWindow({
        width: 1080,
        height: 1920, // Slightly taller to accommodate help button
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        }
    });

    // Automatically grant media permissions when the app asks.
    session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
        if (permission === 'media') {
            return callback(true);
        }
        return callback(false);
    });

    mainWindow.loadFile('index.html');
    // For debugging:
    // mainWindow.webContents.openDevTools();
}

/**
 * Finds the name of the first physical webcam available to DirectShow.
 * It filters out the OBS virtual camera to avoid a feedback loop.
 * @param {function(string|null, Error|null)} callback - The callback to run with the device name.
 */
function findWebcamName(callback) {
    // Determine the path to ffmpeg. When packaged, it's in the resources directory.
    const isPackaged = app.isPackaged;
    const ffmpegPath = isPackaged
        ? path.join(process.resourcesPath, 'app/build-resources/ffmpeg/ffmpeg.exe')
        : path.join(__dirname, 'build-resources/ffmpeg/ffmpeg.exe');

    const command = `"${ffmpegPath}" -list_devices true -f dshow -i dummy`;

    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error executing ffmpeg -list_devices: ${error}`);
            return callback(null, error);
        }
        
        const output = stdout + stderr; // Combine both streams as dshow often logs to stderr
        const lines = output.split('\n');
        const videoDeviceLineIndex = lines.findIndex(line => line.includes('DirectShow video devices'));
        
        if (videoDeviceLineIndex !== -1) {
            // Search for the first video device that is NOT the OBS camera
            for (let i = videoDeviceLineIndex + 1; i < lines.length; i++) {
                if (lines[i].includes('"')) {
                    try {
                        const deviceName = lines[i].match(/"([^"]+)"/)[1];
                        if (!deviceName.toLowerCase().includes('obs')) {
                            console.log(`Found physical webcam: "${deviceName}"`);
                            return callback(deviceName, null);
                        }
                    } catch (e) { /* Ignore lines that don't match */ }
                }
            }
        }
        callback(null, new Error("Could not find a physical webcam."));
    });
}

// --- IPC Listeners (Communication from Renderer to Main) ---

ipcMain.on('start-virtual-camera', (event) => {
    findWebcamName((webcamName, err) => {
        if (err || !webcamName) {
            console.error("Could not start virtual camera, webcam not found.");
            event.sender.send('ffmpeg-error', 'Physical webcam not found.');
            return;
        }

        if (ffmpegProcess) {
            console.log("FFmpeg process is already running.");
            return;
        }

        const isPackaged = app.isPackaged;
        const ffmpegPath = isPackaged
            ? path.join(process.resourcesPath, 'app/build-resources/ffmpeg/ffmpeg.exe')
            : path.join(__dirname, 'build-resources/ffmpeg/ffmpeg.exe');

        const args = [
            '-f', 'dshow',
            '-i', `video=${webcamName}`,
            '-pix_fmt', 'yuv420p',
            '-f', 'dshow',
            'OBS Virtual Camera' // This is the standard name of the OBS driver
        ];

        console.log(`Starting FFmpeg: "${ffmpegPath}" ${args.join(' ')}`);
        
        try {
            ffmpegProcess = spawn(ffmpegPath, args);

            ffmpegProcess.stderr.on('data', (data) => {
                // FFmpeg logs progress to stderr, so we log it for debugging.
                console.error(`ffmpeg: ${data}`);
            });

            ffmpegProcess.on('close', (code) => {
                console.log(`ffmpeg process exited with code ${code}`);
                ffmpegProcess = null;
                // Notify the renderer that ffmpeg has stopped unexpectedly
                event.sender.send('ffmpeg-stopped');
            });
        } catch(spawnError) {
             console.error("Failed to spawn FFmpeg process:", spawnError);
             event.sender.send('ffmpeg-error', 'Failed to start FFmpeg.');
        }
    });
});

ipcMain.on('stop-virtual-camera', () => {
    if (ffmpegProcess) {
        console.log("Stopping FFmpeg process...");
        ffmpegProcess.kill('SIGINT'); // Graceful stop
        ffmpegProcess = null;
    }
});

ipcMain.on('open-quickstart', () => {
    const quickStartPath = app.isPackaged
        ? path.join(process.resourcesPath, 'app', 'QuickStart.html')
        : path.join(__dirname, 'QuickStart.html');
    
    shell.openPath(quickStartPath).then(error => {
        if (error) {
            console.error('Failed to open QuickStart guide:', error);
        }
    });
});

// --- App Lifecycle Events ---

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

// Make sure to clean up the FFmpeg process when the app closes.
app.on('will-quit', () => {
    if (ffmpegProcess) {
        ffmpegProcess.kill();
    }
});