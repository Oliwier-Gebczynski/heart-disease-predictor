CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    age INT NOT NULL,
    blood_pressure INT NOT NULL,
    cholesterol INT NOT NULL,
    smoking BOOLEAN NOT NULL,
    heart_rate INT NOT NULL,
    risk INT NOT NULL
);

INSERT INTO patients (age, blood_pressure, cholesterol, smoking, heart_rate, risk) VALUES
(45, 130, 240, TRUE, 80, 1),
(50, 140, 200, FALSE, 70, 0),
(62, 150, 300, TRUE, 75, 1),
(30, 120, 180, FALSE, 65, 0),
(40, 125, 220, TRUE, 90, 1);
