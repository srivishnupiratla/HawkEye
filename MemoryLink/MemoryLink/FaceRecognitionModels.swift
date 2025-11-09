import Foundation

// MARK: - Face Recognition Models

struct FaceRecognitionResult: Codable, Identifiable {
    let id = UUID()
    let name: String
    let location: FaceLocation
    let confidence: Double
    
    enum CodingKeys: String, CodingKey {
        case name, location, confidence
    }
}

struct FaceLocation: Codable {
    let top: Int
    let right: Int
    let bottom: Int
    let left: Int
}

struct FaceRecognitionResponse: Codable {
    let count: Int
    let faces: [FaceRecognitionResult]
    let error: String?
}

// MARK: - Server Communication Models (Fix for "Cannot find in scope")

// Model for data SENT TO the Python Server
struct ServerRequest: Codable {
    let image: String // Base64 encoded image
    let prompt: String?
    let recognize_faces: Bool
}

// Model for data RECEIVED FROM the Python Server
struct ServerResponse: Codable {
    let response: String?
    let faces: FaceRecognitionResponse?
    let error: String?
}

//
//  FaceRecognitionModels.swift
//  MemoryLink
//
//  Created by Ashwin Subbu on 11/9/25.
//
