;;;; lisp-cognito-philosopher.asd
;;;; ASDF System Definition for Lisp Cognito Philosopher

(asdf:defsystem #:lisp-cognito-philosopher
  :description "A self-modifying, self-referential philosophical reasoning engine in Common Lisp"
  :author "Julian"
  :license "MIT"
  :version "0.1.0"
  :serial t
  :depends-on (#:cl-json
               #:alexandria
               #:trivial-utf-8)
  :components ((:file "package")
               (:file "utils")
               (:file "goedel-encoder")
               (:file "symbolic-operator")
               (:file "geometric-tokenizer")
               (:file "cognito-construct")
               (:file "dialectic-macros")
               (:file "living-logos")
               (:file "reflection-engine")
               (:file "philosopher"))
  :in-order-to ((test-op (test-op #:lisp-cognito-philosopher/tests))))

(asdf:defsystem #:lisp-cognito-philosopher/tests
  :description "Test suite for Lisp Cognito Philosopher"
  :author "Julian"
  :license "MIT"
  :depends-on (#:lisp-cognito-philosopher
               #:fiveam)
  :components ((:module "tests"
                :components ((:file "run-tests")
                             (:file "test-goedel-encoder")
                             (:file "test-symbolic-operator")
                             (:file "test-geometric-tokenizer")
                             (:file "test-dialectic-macros")
                             (:file "test-philosopher"))))
  :perform (test-op (o c) (symbol-call :fiveam '#:run! :lisp-cognito-philosopher-suite)))
