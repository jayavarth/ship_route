#!/bin/bash
 echo "Deployment started..."
 
 # Optional: make sure permissions are good
 chmod -R 755 /var/www/html
 
 # Optional: restart Nginx if needed
 systemctl restart nginx
 
 echo "Deployment finished!"
