-- Database schema for Student Card OCR System

CREATE DATABASE IF NOT EXISTS student_card_db;
USE student_card_db;

CREATE TABLE IF NOT EXISTS students (
    id INT PRIMARY KEY AUTO_INCREMENT,
    mssv VARCHAR(20) UNIQUE NOT NULL,
    ho_ten VARCHAR(100) NOT NULL,
    ngay_sinh DATE,
    nien_khoa VARCHAR(20),
    ngay_het_han DATE,
    avatar_path TEXT,
    face_encoding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_mssv (mssv),
    INDEX idx_ho_ten (ho_ten)
);

