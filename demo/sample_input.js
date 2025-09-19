function saveUserProfile(profileData, callback) {
    database.update('users', profileData.id, profileData, (err, result) => {
        if (err) {
            callback(err);
        } else {
            callback(null, result);
        }
    });
}