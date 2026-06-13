# An Introduction to HTTP

HTTP, the Hypertext Transfer Protocol, is the foundation of data communication
on the World Wide Web. It is a request-response protocol: a client, usually a
web browser, sends a request to a server, and the server returns a response.

An HTTP request includes a method that describes the desired action. The most
common methods are GET, which retrieves a resource, and POST, which submits data
to the server. Other methods include PUT, DELETE, and HEAD.

Every HTTP response carries a status code that tells the client what happened.
Codes in the 200 range mean success, codes in the 300 range mean redirection,
codes in the 400 range indicate a client error such as 404 Not Found, and codes
in the 500 range indicate a server error.

HTTP is stateless, meaning each request is independent and the server does not
remember previous requests by default. To preserve state across requests, web
applications use mechanisms such as cookies and tokens. The secure version of
the protocol, HTTPS, encrypts the connection using TLS.
