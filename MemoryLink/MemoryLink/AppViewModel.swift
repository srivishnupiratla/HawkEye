import Foundation
import SwiftUI
import Combine
import AVFoundation
import AudioToolbox // For the sound effect
import UIKit // Needed for UIImage in CVPixelBuffer extension

@MainActor
class AppViewModel: ObservableObject {
    
    // MARK: - Published Properties
    @Published var statusText: String = "Disconnected"
    @Published var isStreaming: Bool = false
    @Published var isListening: Bool = false
    @Published var isCameraReady: Bool = false
    @Published var recognizedFaces: [FaceRecognitionResult] = []
    @Published var isFaceRecognitionEnabled: Bool = false
    
    // MARK: - Private State
    private var lastSentFrameTime: TimeInterval = 0
    private var latestPixelBuffer: CVPixelBuffer? // Cache the last frame
    
    // MARK: - Services
    private let webSocketService = WebSocketService()
    private let cameraService = CameraService()
    private let speechService = SpeechService()
    private let locationService = LocationService()
    
    var cameraSession: AVCaptureSession {
        cameraService.session
    }
    
    init() {
        webSocketService.delegate = self
        cameraService.delegate = self
        speechService.delegate = self
        locationService.delegate = self
    }
    
    // MARK: - Public Intents
    
    func connect() {
        guard !isStreaming else { return }
        statusText = "Connecting..."
        webSocketService.connect()
        cameraService.startCapture()
        locationService.startMonitoring()
    }
    
    func disconnect() {
        guard isStreaming else { return }
        webSocketService.disconnect()
        cameraService.stopCapture()
        speechService.stopListening()
        locationService.stopMonitoring()
        isCameraReady = false
        isListening = false
        recognizedFaces = []
        isFaceRecognitionEnabled = false
    }
    
    func toggleFaceRecognition() {
        isFaceRecognitionEnabled.toggle()
        
        if isFaceRecognitionEnabled {
            statusText = "Face recognition enabled"
            print("✅ Face recognition toggled ON - will scan continuously")
            // Immediately trigger face recognition
            recognizeFace()
        } else {
            statusText = "Face recognition disabled"
            print("❌ Face recognition toggled OFF")
            recognizedFaces = []
        }
    }
    
    func recognizeFace() {
        guard isStreaming else {
            print("⚠️ Not streaming, cannot recognize face")
            return
        }
        guard let pixelBuffer = latestPixelBuffer else {
            print("⚠️ No frame available for face recognition")
            statusText = "No frame available"
            return
        }
        
        statusText = "Recognizing faces..."
        print("🔍 IMMEDIATE face recognition requested")
        
        // Convert frame directly and send immediately, bypassing all throttling
        guard let base64Image = pixelBuffer.toBase64String() else {
            print("❌ Failed to convert pixel buffer to Base64")
            statusText = "Frame conversion failed"
            return
        }
        
        let request = ServerRequest(
            image: base64Image,
            prompt: "Analyze this frame according to the system instructions.",
            recognize_faces: true
        )
        
        webSocketService.send(request: request)
        print("✅ Face recognition request sent immediately")
    }
    
    func trainFace(name: String) {
        statusText = "To train: Add '\(name).jpg' to server's known_faces/ folder"
        // The server needs to be updated to handle saving frames for training.
        // For now, this just updates the status text.
        print("Train face: Add '\(name).jpg' to server's known_faces/ folder")
        
        // Clear the message after a delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
            if self.isStreaming {
                self.statusText = "Listening for 'starfish'..."
            }
        }
    }
    
    // MARK: - Consolidated Send Function
    
    /// This is the new central function for all server communication.
    private func sendRequest(prompt: String?, forceFaceRec: Bool = false) {
        guard isStreaming else {
            print("Not streaming, request cancelled.")
            return
        }
        
        guard let pixelBuffer = latestPixelBuffer else {
            print("No frame cached, request cancelled. Waiting for camera.")
            return
        }
        
        // Convert CVPixelBuffer to Base64 String
        guard let base64Image = pixelBuffer.toBase64String() else {
            print("Failed to convert pixel buffer to Base64")
            return
        }
        
        let recognize = isFaceRecognitionEnabled || forceFaceRec
        let promptText = prompt ?? "Analyze this frame according to the system instructions."
        
        // Create the single request object the server expects
        let request = ServerRequest(
            image: base64Image,
            prompt: promptText,
            recognize_faces: recognize
        )
        
        webSocketService.send(request: request)
    }
}

// MARK: - WebSocketDelegate
extension AppViewModel: WebSocketDelegate {
    func webSocketDidConnect() {
        statusText = "Listening for 'starfish'..."
        isStreaming = true
        isListening = true
        speechService.startListeningForTrigger()
    }
    
    func webSocketDidDisconnect() {
        statusText = "Error: Disconnected"
        isStreaming = false
        isListening = false
        recognizedFaces = []
    }
    
    func webSocketDidReceiveAnswer(_ text: String) {
        print("📨 Received server message: \(text)")
        // Don't update status with server responses - keep it clean
        // Status is managed by user actions and face recognition results only
    }
    
    func webSocketDidReceiveFaceRecognition(_ faces: [FaceRecognitionResult]) {
        DispatchQueue.main.async {
            self.recognizedFaces = faces
            
            if !faces.isEmpty {
                let names = faces.map { $0.name }.joined(separator: ", ")
                self.statusText = "Detected: \(names)"
                print("👤 Faces detected: \(names)")
                
                // Ding sound
                AudioServicesPlaySystemSound(SystemSoundID(1054))
                
                // Speak result
                if let first = faces.first {
                    if first.name != "Unknown" {
                        TextToSpeechService.shared.speak("This is \(first.name)")
                    } else {
                        // Optional:
                        TextToSpeechService.shared.speak("I don't know who this is")
                    }
                }
            } else {
                self.statusText = "No faces detected"
            }
        }
    }
}

// MARK: - CameraServiceDelegate
extension AppViewModel: CameraServiceDelegate {
    func cameraIsReady() {
        isCameraReady = true
    }
    
    func cameraDidCaptureFrame(_ pixelBuffer: CVPixelBuffer) {
        // 1. Always cache the latest frame
        self.latestPixelBuffer = pixelBuffer
        
        // 2. Throttle sending to match server processing interval (2 seconds)
        let now = Date().timeIntervalSince1970
        if now - lastSentFrameTime < 2.0 { return } // Send frames every 2 seconds
        
        // 3. ALWAYS send frames for the LLM to analyze
        // Face recognition is controlled by the flag, but frames are always sent
        lastSentFrameTime = now
        
        if isFaceRecognitionEnabled {
            print("📸 Sending frame with continuous face recognition")
            sendRequest(prompt: nil, forceFaceRec: true)
        } else {
            print("📸 Sending frame for LLM analysis only")
            sendRequest(prompt: nil, forceFaceRec: false)
        }
    }
}

// MARK: - SpeechServiceDelegate
extension AppViewModel: SpeechServiceDelegate {
    
    func speechDidFinishListening(with text: String) {
        print("✅ Got final text: '\(text)'")
        
        if !text.isEmpty {
            AudioServicesPlaySystemSound(SystemSoundID(1054))
            statusText = "Processing: \(text)"
            
            // Check if the prompt is asking "who" to force face recognition.
            let forceFaceRec = text.lowercased().contains("who") ||
                               text.lowercased().contains("identify") ||
                               text.lowercased().contains("recognize")
                               
            sendRequest(prompt: text, forceFaceRec: forceFaceRec)
            
            // Restart trigger listening
            if isStreaming {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    self.statusText = "Listening for 'starfish'..."
                    self.isListening = true
                    self.speechService.startListeningForTrigger()
                }
            } else {
                isListening = false
            }
        } else {
            // No speech detected
            statusText = "No speech detected"
            if isStreaming {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    self.statusText = "Listening for 'starfish'..."
                    self.isListening = true
                    self.speechService.startListeningForTrigger()
                }
            } else {
                isListening = false
            }
        }
    }
    
    func speechDidFail(with error: String) {
        print("❌ Speech failed: \(error)")
        statusText = "Error: \(error)"
        
        if isStreaming {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                self.statusText = "Listening for 'starfish'..."
                self.isListening = true
                self.speechService.startListeningForTrigger()
            }
        } else {
            isListening = false
        }
    }
    
    func triggerWordDetected() {
        print("🌟 Trigger word detected - listening for command")
        guard isStreaming else { return }
        statusText = "Say your command..."
        isListening = true
    }
    
    func sendPromptToAPI(_ prompt: String) {
        print("📤 Sending prompt to API: \(prompt)")
        sendRequest(prompt: prompt)
    }
}

// MARK: - LocationServiceDelegate
extension AppViewModel: LocationServiceDelegate {
    func didExitHomeRegion() {
        guard isStreaming else { return }
        statusText = "Geofence exited. Checking for danger..."
        
        // Send a request with the geofence prompt and the latest frame
        sendRequest(prompt: "The user has left their home geofence. Check if there are any potential dangers in the current frame.", forceFaceRec: false)
    }
}

extension CVPixelBuffer {
    func toBase64String(compressionQuality: CGFloat = 1.0) -> String? {
        let ciImage = CIImage(cvPixelBuffer: self)
        let context = CIContext()
        let rotated = ciImage.oriented(.right)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        // Build the CIImageRepresentationOption key for lossy compression quality
        let qualityKey = CIImageRepresentationOption(
            rawValue: kCGImageDestinationLossyCompressionQuality as String
        )
        
        let options: [CIImageRepresentationOption: Any] = [
            qualityKey: compressionQuality
        ]
        
        guard let jpegData = context.jpegRepresentation(
            of: rotated,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            options: [:]
        ) else {
            print("ERROR: Failed to create JPEG representation from CIImage.")
            return nil
        }
        
        print("DEBUG: Successfully generated JPEG data (\(jpegData.count) bytes).")
        return jpegData.base64EncodedString()
    }
}
