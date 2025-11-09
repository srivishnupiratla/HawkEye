import Foundation
import AVFoundation

protocol CameraServiceDelegate: AnyObject {
    // --- THIS IS THE NEW LINE ---
    func cameraIsReady() // Tells the view model the session is configured
    
    func cameraDidCaptureFrame(_ pixelBuffer: CVPixelBuffer)
}

/// Captures and processes video frames from the back camera.
class CameraService: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    let session = AVCaptureSession()
    
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.memorylink.sessionQueue")
    
    weak var delegate: CameraServiceDelegate?
    
    private var lastFrameTime = CMTime.zero
    private let frameInterval = CMTimeMake(value: 1, timescale: 5)
    private var isSetup = false
    
    override init() {
        super.init()
    }
    
    private func setupCaptureSession(completion: @escaping (Bool) -> Void) {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let videoInput = try? AVCaptureDeviceInput(device: videoDevice) else {
            print("Error: Could not find back camera or create input.")
            session.commitConfiguration()
            completion(false)
            return
        }
        
        if session.canAddInput(videoInput) {
            session.addInput(videoInput)
        }
        
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "com.memorylink.videoQueue"))
        
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        
        session.commitConfiguration()
        isSetup = true
        completion(true)
    }
    
    func startCapture() {
        sessionQueue.async {
            if self.session.isRunning { return }
            
            if !self.isSetup {
                self.setupCaptureSession { [weak self] success in
                    guard success, let self = self else { return }
                    self.session.startRunning()
                    
                    // --- THIS IS THE FIX ---
                    // Tell the delegate (AppViewModel) that we are 100% ready
                    DispatchQueue.main.async {
                        self.delegate?.cameraIsReady()
                    }
                }
            } else {
                self.session.startRunning()
                // --- ALSO FIX HERE ---
                // If we are already set up, just tell the delegate
                DispatchQueue.main.async {
                    self.delegate?.cameraIsReady()
                }
            }
        }
    }
    
    func stopCapture() {
        sessionQueue.async {
            if self.session.isRunning {
                self.session.stopRunning()
            }
        }
    }
    
    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let currentTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        if (currentTime - lastFrameTime) < frameInterval { return }
        lastFrameTime = currentTime
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        DispatchQueue.main.async {
            self.delegate?.cameraDidCaptureFrame(pixelBuffer)
        }
    }
}
