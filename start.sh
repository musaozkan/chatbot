#!/bin/bash

echo "Please select a command to run:"
echo "1. Train and chatbot" 
echo "2. Chatbot"
echo "3. Exit"

read choice

case $choice in
  1)
    python ./training.py 
    python ./chatbot.py
    ;;
  2)
    python ./chatbot.py

    ;;
  3)
    echo "Exiting..."
    exit 0
    ;;
  *)
    echo "Invalid choice"
    exit 1
    ;;
esac