import SwiftUI
import AVFoundation

struct ContentView: View {
    
    @StateObject private var viewModel = AppViewModel()
    @State private var nameForTraining: String = ""

    var body: some View {
        ZStack {
            // Black background
            Color.black.ignoresSafeArea()
            
            // Show camera only when ready
            if viewModel.isCameraReady {
                CameraPreview(session: viewModel.cameraSession)
                    .ignoresSafeArea()
                
                // Face recognition overlay
                if viewModel.isFaceRecognitionEnabled && !viewModel.recognizedFaces.isEmpty {
                    FaceOverlayView(
                        faces: viewModel.recognizedFaces,
                        frameSize: CGSize(width: 640, height: 480)
                    )
                    .ignoresSafeArea()
                }
            }
            
            VStack {
                // Status + face icon
                HStack {
                    Text(viewModel.statusText)
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                    
                    if viewModel.isFaceRecognitionEnabled {
                        Image(systemName: "face.smiling.fill")
                            .foregroundColor(.white)
                            .font(.title2)
                    }
                }
                .padding()
                .background(statusColor())
                .cornerRadius(12)
                .animation(.easeInOut, value: viewModel.statusText)
                
                Spacer()
                
                // Detected faces list
                if !viewModel.recognizedFaces.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Detected Faces:")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding(.horizontal)
                        
                        ForEach(Array(viewModel.recognizedFaces.enumerated()), id: \.offset) { _, face in
                            HStack {
                                Image(systemName: face.name == "Unknown" ? "person.fill.questionmark" : "person.fill.checkmark")
                                    .foregroundColor(face.name == "Unknown" ? .red : .green)
                                
                                Text(face.name)
                                    .font(.system(size: 16, weight: .semibold))
                                    .foregroundColor(.white)
                                
                                if face.name != "Unknown" {
                                    Text("\(Int(face.confidence * 100))%")
                                        .font(.system(size: 14))
                                        .foregroundColor(.white.opacity(0.7))
                                }
                                
                                Spacer()
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(Color.black.opacity(0.7))
                            )
                        }
                    }
                    .padding(.horizontal)
                }
                
                // Main buttons
                VStack(spacing: 16) {
                    if !viewModel.isStreaming {
                        Button(action: viewModel.connect) {
                            Text("Connect")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .font(.title)
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                                .background(Color.blue)
                                .cornerRadius(16)
                        }
                    } else {
                        Button(action: viewModel.toggleFaceRecognition) {
                            HStack {
                                Image(systemName: viewModel.isFaceRecognitionEnabled ? "face.smiling.fill" : "face.smiling")
                                Text(viewModel.isFaceRecognitionEnabled ? "Disable Face Recognition" : "Enable Face Recognition")
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .font(.headline)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                            .background(viewModel.isFaceRecognitionEnabled ? Color.green : Color.purple)
                            .cornerRadius(16)
                        }
                        
                        HStack(spacing: 16) {
                            Button(action: viewModel.recognizeFace) {
                                Text("Who is this?")
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .font(.headline)
                                    .fontWeight(.bold)
                                    .foregroundColor(.white)
                                    .background(Color.orange)
                                    .cornerRadius(16)
                            }
                            
                            Button(action: trainFace) {
                                Text("Train Face")
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .font(.headline)
                                    .fontWeight(.bold)
                                    .foregroundColor(.black)
                                    .background(Color.yellow)
                                    .cornerRadius(16)
                            }
                        }
                        
                        Button(action: viewModel.disconnect) {
                            Text("Disconnect")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .font(.headline)
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                                .background(Color.gray)
                                .cornerRadius(16)
                        }
                    }
                }
                .padding()
            }
            .padding()
        }
        .onAppear {
            viewModel.connect()
        }
    }
    
    func statusColor() -> Color {
        if viewModel.statusText.lowercased().contains("error") {
            return .red
        }
        if viewModel.isListening {
            return .green
        }
        if viewModel.isStreaming {
            return .blue.opacity(0.8)
        }
        return .gray.opacity(0.8)
    }
    
    func trainFace() {
        let alert = UIAlertController(title: "Train Face", message: "Enter the name of the person.", preferredStyle: .alert)
        alert.addTextField { textField in
            textField.placeholder = "e.g., John Smith"
        }
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        alert.addAction(UIAlertAction(title: "Train", style: .default, handler: { _ in
            if let name = alert.textFields?.first?.text, !name.isEmpty {
                viewModel.trainFace(name: name)
            }
        }))
        
        if let rootVC = UIApplication.shared.windows.first?.rootViewController {
            rootVC.present(alert, animated: true)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
