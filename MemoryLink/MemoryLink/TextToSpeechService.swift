import AVFoundation

final class TextToSpeechService {
    static let shared = TextToSpeechService()
    private let synthesizer = AVSpeechSynthesizer()
    
    private init() {}
    
    func speak(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        
        DispatchQueue.main.async {
            // Ensure audio session allows output
            let session = AVAudioSession.sharedInstance()
            do {
                try session.setCategory(.playAndRecord,
                                        mode: .default,
                                        options: [.defaultToSpeaker, .mixWithOthers])
                try session.setActive(true, options: .notifyOthersOnDeactivation)
            } catch {
                print("🔊 Audio session for TTS error: \(error)")
            }
            
            if self.synthesizer.isSpeaking {
                self.synthesizer.stopSpeaking(at: .immediate)
            }
            
            let utterance = AVSpeechUtterance(string: trimmed)
            utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
            utterance.rate = AVSpeechUtteranceDefaultSpeechRate
            utterance.pitchMultiplier = 1.0
            
            self.synthesizer.speak(utterance)
        }
    }
}
