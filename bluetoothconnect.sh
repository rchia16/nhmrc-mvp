RAZER_MAC = "28:FA:19:0F:E9:CF"

echo "Attempted to connect to Razer Anzu Smart Glasses"

sudo systemctl start bluetooth
sleep 2

echo -e "power on\nconnect $RAZER_MAC\nquit" | bluetoothctl
echo -e "info\n" | bluetoothctl

echo "Connection script finished."
