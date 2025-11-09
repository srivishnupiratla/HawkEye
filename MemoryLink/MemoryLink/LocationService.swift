import Foundation
import CoreLocation

protocol LocationServiceDelegate: AnyObject {
    func didExitHomeRegion()
}

/// Manages geofencing for the "Home" region.
class LocationService: NSObject, CLLocationManagerDelegate {
    
    private let locationManager = CLLocationManager()
    weak var delegate: LocationServiceDelegate?
    
    private var homeRegion: CLCircularRegion?
    
    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.requestAlwaysAuthorization()
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
    }
    
    func startMonitoring() {
        guard CLLocationManager.isMonitoringAvailable(for: CLCircularRegion.self) else {
            print("Geofencing is not supported on this device.")
            return
        }
        
        // Use the first location as "Home"
        if homeRegion == nil, let location = locationManager.location {
            setupHomeGeofence(at: location.coordinate)
        }
        
        locationManager.startUpdatingLocation()
    }
    
    func stopMonitoring() {
        if let region = homeRegion {
            locationManager.stopMonitoring(for: region)
        }
        locationManager.stopUpdatingLocation()
    }
    
    private func setupHomeGeofence(at coordinate: CLLocationCoordinate2D) {
        let region = CLCircularRegion(center: coordinate, radius: 100, identifier: "Home") // 100-meter radius
        region.notifyOnExit = true
        region.notifyOnEntry = false
        
        self.homeRegion = region
        locationManager.startMonitoring(for: region)
        print("Geofence set up at: \(coordinate.latitude), \(coordinate.longitude)")
    }
    
    // MARK: - CLLocationManagerDelegate
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        // Set up the geofence on the first location update if it's not set
        if homeRegion == nil, let firstLocation = locations.first {
            setupHomeGeofence(at: firstLocation.coordinate)
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didExitRegion region: CLRegion) {
        if region.identifier == "Home" {
            print("User exited 'Home' region.")
            DispatchQueue.main.async {
                self.delegate?.didExitHomeRegion()
            }
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didChangeAuthorization status: CLAuthorizationStatus) {
        if status == .authorizedAlways {
            locationManager.startUpdatingLocation()
        } else {
            print("Location 'Always' authorization is required for geofencing.")
        }
    }
    
    func locationManager(_ manager: CLLocationManager, monitoringDidFailFor region: CLRegion?, withError error: Error) {
        print("Geofence monitoring failed: \(error.localizedDescription)")
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location manager failed: \(error.localizedDescription)")
    }
}
