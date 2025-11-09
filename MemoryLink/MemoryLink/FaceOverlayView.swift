import SwiftUI

struct FaceOverlayView: View {
    let faces: [FaceRecognitionResult]
    let frameSize: CGSize
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                ForEach(faces) { face in
                    FaceBox(face: face, frameSize: frameSize, viewSize: geometry.size)
                }
            }
        }
    }
}

struct FaceBox: View {
    let face: FaceRecognitionResult
    let frameSize: CGSize
    let viewSize: CGSize
    
    var body: some View {
        let rect = convertToViewCoordinates(face.location)
        
        ZStack(alignment: .top) {
            // Face bounding box
            Rectangle()
                .stroke(face.name == "Unknown" ? Color.red : Color.green, lineWidth: 3)
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)
            
            // Name label
            if face.name != "Unknown" {
                Text(face.name)
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.green)
                    )
                    .position(x: rect.midX, y: rect.minY - 15)
            } else {
                Text("Unknown")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.red)
                    )
                    .position(x: rect.midX, y: rect.minY - 15)
            }
        }
    }
    
    private func convertToViewCoordinates(_ location: FaceLocation) -> CGRect {
        // Convert from image coordinates to view coordinates
        let scaleX = viewSize.width / frameSize.width
        let scaleY = viewSize.height / frameSize.height
        
        let x = CGFloat(location.left) * scaleX
        let y = CGFloat(location.top) * scaleY
        let width = CGFloat(location.right - location.left) * scaleX
        let height = CGFloat(location.bottom - location.top) * scaleY
        
        return CGRect(x: x, y: y, width: width, height: height)
    }
}

// Preview helper
struct FaceOverlayView_Previews: PreviewProvider {
    static var previews: some View {
        FaceOverlayView(
            faces: [
                FaceRecognitionResult(
                    name: "John Doe",
                    location: FaceLocation(top: 100, right: 300, bottom: 300, left: 100),
                    confidence: 0.95
                )
            ],
            frameSize: CGSize(width: 640, height: 480)
        )
        .frame(width: 400, height: 300)
        .background(Color.black)
    }
}
