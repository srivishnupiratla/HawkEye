import SwiftUI

struct ContentView: View {
    
    @StateObject private var viewModel = AppViewModel()
    @State private var nameForTraining: String = ""

    var body: some View {
        ZStack {
            // --- THIS IS THE CHANGE ---
            // Instead of Color.black, we show the live camera feed.
            // It will fill the whole screen.
            CameraPreview()
                .edgesIgnoringSafeArea(.all)
            // --- END OF CHANGE ---
            
            VStack {
                // 1. Status Indicator
                Text(viewModel.statusText)
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                    .padding()
                    .background(statusColor())
                    .cornerRadius(12)
                    .animation(.easeInOut, value: viewModel.statusText)
                
                Spacer()
                
                // 2. Main Control Buttons
                VStack(spacing: 16) {
                    if !viewModel.isStreaming {
                        // Connect Button
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
                        // Ask Query Button
                        Button(action: viewModel.startQuery) {
                            Text(viewModel.isListening ? "Listening..." : "Ask Query")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .font(.title)
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                                .background(Color.green)
                                .cornerRadius(16)
                        }
                        .disabled(viewModel.isListening)
                        
                        // Facial Recognition Buttons
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
                        
                        // Disconnect Button
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
            // Request permissions as soon as the view appears
            viewModel.connect()
        }
    }
    
    /// Helper for a dynamic status color
    func statusColor() -> Color {
        if viewModel.statusText.lowercased().contains("error") {
            return .red
        }
        if viewModel.statusText.lowercased().contains("listening") {
            return .green
        }
        if viewModel.isStreaming {
            return .blue.opacity(0.8)
        }
        return .gray.opacity(0.8)
    }
    
    /// Shows an alert to get the name for training
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
        
        // Find the root view controller to present the alert
        if let rootVC = UIApplication.shared.windows.first?.rootViewController {
            rootVC.present(alert, animated: true)
        }
    }
}