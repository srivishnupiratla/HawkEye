import Foundation
import AVFoundation
import UIKit

// MARK: - WebSocketDelegate Protocol
protocol WebSocketDelegate: AnyObject {
    func webSocketDidConnect()
    func webSocketDidDisconnect()
    func webSocketDidReceiveAnswer(_ text: String)
    func webSocketDidReceiveFaceRecognition(_ faces: [FaceRecognitionResult])
}

// MARK: - WebSocketService
class WebSocketService: NSObject, URLSessionWebSocketDelegate {
    
    weak var delegate: WebSocketDelegate?
    
    private var webSocketTask: URLSessionWebSocketTask?
    
    // Replace with your machine's IP, e.g. ws://192.168.1.10:8000/ws
    //Surya: 172.31.42.206
    //Ashwin: 206.168.83.194
    private let serverURL = URL(string: "ws://206.168.83.194:8000/ws")!
    
    private var isConnected = false
    
    override init() {
        super.init()
    }
    
    func connect() {
        print("Connecting to WebSocket...")
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: OperationQueue())
        webSocketTask = session.webSocketTask(with: serverURL)
        webSocketTask?.resume()
        receiveMessage()
    }
    
    func disconnect() {
        print("Disconnecting from WebSocket...")
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        isConnected = false
    }
    
    // MARK: - Sending Data
    
    func send(request: ServerRequest) {
        guard isConnected else {
            print("Cannot send, WebSocket is not connected.")
            return
        }
        
        do {
            let data = try JSONEncoder().encode(request)
            guard let jsonString = String(data: data, encoding: .utf8) else {
                print("Failed to encode request to JSON string")
                return
            }
            
            webSocketTask?.send(.string(jsonString)) { error in
                if let error = error {
                    print("WebSocket send error: \(error.localizedDescription)")
                }
            }
        } catch {
            print("Failed to encode ServerRequest: \(error)")
        }
    }
    
    // MARK: - Receiving Data
    
    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .success(let message):
                switch message {
                case .string(let text):
                    self.handleReceivedText(text)
                case .data(let data):
                    print("Received unexpected binary data: \(data)")
                @unknown default:
                    break
                }
                self.receiveMessage()
                
            case .failure(let error):
                print("WebSocket receive error: \(error.localizedDescription)")
                self.handleDisconnect()
            }
        }
    }
    
    private func handleReceivedText(_ text: String) {
        guard let data = text.data(using: .utf8) else { return }
        
        do {
            let response = try JSONDecoder().decode(ServerResponse.self, from: data)
            
            DispatchQueue.main.async {
                if let responseText = response.response {
                    self.delegate?.webSocketDidReceiveAnswer(responseText)
                }
                
                if let faceData = response.faces {
                    self.delegate?.webSocketDidReceiveFaceRecognition(faceData.faces)
                }
                
                if let error = response.error {
                    print("Server error: \(error)")
                }
            }
            
        } catch {
            print("Failed to decode server response: \(error)")
            print("Raw response: \(text)")
        }
    }
    
    private func handleDisconnect() {
        if isConnected {
            isConnected = false
            DispatchQueue.main.async {
                self.delegate?.webSocketDidDisconnect()
            }
        }
    }
    
    // MARK: - URLSessionWebSocketDelegate
    
    func urlSession(_ session: URLSession,
                    webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol `protocol`: String?) {
        print("✅ WebSocket did connect")
        DispatchQueue.main.async {
            self.isConnected = true
            self.delegate?.webSocketDidConnect()
        }
    }
    
    func urlSession(_ session: URLSession,
                    webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
                    reason: Data?) {
        print("❌ WebSocket did disconnect with code: \(closeCode.rawValue)")
        handleDisconnect()
    }
}
