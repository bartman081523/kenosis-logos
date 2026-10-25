;;;; tests/test-philosopher.lisp
;;;; Integration tests for the philosopher system

(in-package #:lisp-cognito-philosopher-tests)

(test philosopher-creation
  "Test philosopher creation"
  (let ((p (make-philosopher "Test problem")))
    (is (not (null p)))
    (is (cognito-construct-p p))
    (is (string= (cognito-construct-problem p) "Test problem"))))

(test basic-thought-loop
  "Test basic thought loop"
  (let ((p (make-philosopher "What is consciousness?")))
    ;; Think
    (think p "consciousness")
    (is (> (cognito-construct-thought-count p) 0))
    
    ;; Reflect
    (let ((reflection (reflect p)))
      (is (getf reflection :gap))
      (is (numberp (getf reflection :gap))))
    
    ;; Output
    (let ((output-str (output p)))
      (is (stringp output-str))
      (is (> (length output-str) 0)))))

(test goedel-gap-tracking
  "Test Gödel Gap tracking"
  (let ((p (make-philosopher "Test")))
    (dotimes (i 5)
      (think p "test")
      (reflect p))
    (let ((gap (get-goedel-gap p)))
      (is (numberp gap))
      (is (>= gap 0.0))
      (is (<= gap 1.0)))
    (let ((avg-gap (get-average-gap p)))
      (is (numberp avg-gap)))))

(test self-correction
  "Test self-correction mechanism"
  (let ((p (make-philosopher "Test")))
    ;; Create high gap situation
    (dotimes (i 3)
      (reflect p))
    ;; Apply self-correction
    (let ((corrected (self-correct p)))
      (is (or (eq corrected t) (eq corrected nil))))))

(test living-constructs
  "Test living constructs"
  (let ((emergence (make-living-construct :emergence))
        (consciousness (make-living-construct :consciousness)))
    (is (not (null emergence)))
    (is (not (null consciousness)))
    
    ;; Let them live
    (let ((val1 (live emergence '((x . 1.0))))
          (val2 (live consciousness '((x . 1.0)))))
      (is (numberp val1))
      (is (numberp val2)))
    
    ;; Express
    (let ((expr1 (express emergence))
          (expr2 (express consciousness)))
      (is (stringp expr1))
      (is (stringp expr2)))))
