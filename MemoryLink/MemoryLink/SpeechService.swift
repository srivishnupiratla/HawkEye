import Foundation
import AVFoundation
import Speech
import Combine

protocol SpeechServiceDelegate: AnyObject {
    func speechDidFinishListening(with text: String)
    func speechDidFail(with error: String)
    func triggerWordDetected()
}

@MainActor
class SpeechService: NSObject {
    
    // MARK: - Public
    weak var delegate: SpeechServiceDelegate?
    
    // MARK: - Private
    private let audioEngine = AVAudioEngine()
    private var recognitionTask: SFSpeechRecognitionTask?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var isDynamicListening = false
    private var lastSpeechTime: Date?
    private var silenceTimer: Timer?
    private let silenceTimeout: TimeInterval = 2.5
    private var triggerDetected = false
    private var accumulatedText = ""
    
    // MARK: - Trigger Listener
    
    func startListeningForTrigger() {
        print("🎤 Starting trigger word listening")
        isDynamicListening = false
        triggerDetected = false
        accumulatedText = ""
        lastSpeechTime = nil
        
        startRecognition { [weak self] text, isFinal in
            guard let self = self, !self.triggerDetected else { return }
            
            // Check if trigger word is in the text
            if text.lowercased().contains("starfish") {
                print("✅ Trigger word 'starfish' detected!")
                self.triggerDetected = true
                
                // Play ding immediately
                AudioServicesPlaySystemSound(SystemSoundID(1057))
                
                // Notify delegate
                self.delegate?.triggerWordDetected()
                
                // Transition to dynamic mode while keeping recognition running
                self.transitionToDynamicMode()
            }
        }
    }
    
    // MARK: - Dynamic Listening
    
    private func transitionToDynamicMode() {
        print("🔄 Transitioning to dynamic listening mode")
        isDynamicListening = true
        accumulatedText = ""
        lastSpeechTime = Date()
        
        // Start silence detection
        startSilenceDetection()
    }
    
    func startDynamicListening() {
        // This is now called from the delegate after the ding
        // We're already in dynamic mode, just continue
        print("▶️ Dynamic listening confirmed")
    }
    
    // MARK: - Silence Detection
    
    private func startSilenceDetection() {
        silenceTimer?.invalidate()
        
        silenceTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self, self.isDynamicListening else {
                self?.silenceTimer?.invalidate()
                return
            }
            
            guard let lastSpeech = self.lastSpeechTime else { return }
            
            let elapsed = Date().timeIntervalSince(lastSpeech)
            if elapsed >= self.silenceTimeout {
                print("🔇 Silence detected (\(String(format: "%.1f", elapsed))s)")
                self.silenceTimer?.invalidate()
                self.finishDynamicListening()
            }
        }
    }
    
    private func finishDynamicListening() {
        guard isDynamicListening else { return }
        
        print("✋ Finishing dynamic listening with text: '\(accumulatedText)'")
        let finalText = accumulatedText
        
        stopRecognition()
        
        // Notify delegate with the accumulated text
        delegate?.speechDidFinishListening(with: finalText)
    }
    
    // MARK: - Core Recognition Logic
    
    private func startRecognition(handler: @escaping (String, Bool) -> Void) {
        // Cancel any existing task
        if recognitionTask != nil {
            recognitionTask?.cancel()
            recognitionTask = nil
        }
        
        // End any existing request
        if recognitionRequest != nil {
            recognitionRequest?.endAudio()
            recognitionRequest = nil
        }
        
        // Stop audio engine if running
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        
        // Small delay to ensure cleanup
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            guard let self = self else { return }
            self.setupAndStartRecognition(handler: handler)
        }
    }
    
    private func setupAndStartRecognition(handler: @escaping (String, Bool) -> Void) {
        // Setup audio session
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .measurement, options: .duckOthers)
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("❌ Audio session error: \(error)")
            delegate?.speechDidFail(with: "Audio session error: \(error.localizedDescription)")
            return
        }
        
        // Create new request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            print("❌ Failed to create recognition request")
            return
        }
        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.requiresOnDeviceRecognition = false
        
        // Setup audio tap
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.removeTap(onBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }
        
        // Start audio engine
        audioEngine.prepare()
        do {
            try audioEngine.start()
            print("🎙️ Audio engine started")
        } catch {
            print("❌ Audio engine error: \(error)")
            delegate?.speechDidFail(with: "Audio engine error: \(error.localizedDescription)")
            return
        }
        
        // Start recognition task
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            
            if let result = result {
                let text = result.bestTranscription.formattedString
                let isFinal = result.isFinal
                
                // If we're in dynamic mode, accumulate text AFTER the trigger word
                if self.isDynamicListening && self.triggerDetected {
                    // Remove the trigger word and anything before it
                    let cleanedText = self.removeTextBeforeTrigger(text)
                    if !cleanedText.isEmpty {
                        self.accumulatedText = cleanedText
                        self.lastSpeechTime = Date()
                        print("📝 Accumulated: '\(cleanedText)'")
                    }
                }
                
                // Call the handler for trigger detection
                handler(text, isFinal)
                
                // If final and we're NOT in dynamic mode yet, restart
                if isFinal && !self.isDynamicListening {
                    print("🔁 Final result in trigger mode, restarting")
                    self.stopRecognition()
                }
            }
            
            if let error = error {
                let nsError = error as NSError
                
                // Ignore expected errors (don't even log them)
                if nsError.domain == "kAFAssistantErrorDomain" && (nsError.code == 1110 || nsError.code == 216) {
                    // "No speech detected" or "Cancelled" - these are normal
                    return
                }
                
                if nsError.domain == "kLSRErrorDomain" && nsError.code == 301 {
                    // "Cancelled" - normal during restart
                    return
                }
                
                // Only log and report unexpected errors
                print("⚠️ Recognition error: \(error)")
                self.delegate?.speechDidFail(with: "Recognition error: \(error.localizedDescription)")
            }
        }
    }
    
    private func removeTextBeforeTrigger(_ text: String) -> String {
        let lowercased = text.lowercased()
        if let range = lowercased.range(of: "starfish") {
            let afterTrigger = String(text[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            return afterTrigger
        }
        return text
    }
    
    func stopListening() {
        print("🛑 Stop listening called")
        stopRecognition()
    }
    
    private func stopRecognition() {
        silenceTimer?.invalidate()
        silenceTimer = nil
        
        // Stop audio engine
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        
        // Cancel recognition task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // End recognition request
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        
        // Deactivate audio session
        do {
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            print("⚠️ Error deactivating audio session: \(error)")
        }
        
        isDynamicListening = false
        triggerDetected = false
    }
}
