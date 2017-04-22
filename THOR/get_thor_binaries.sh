mkdir -p thor_binary
cd thor_binary
wget https://s3-us-west-2.amazonaws.com/ai2-vision-robosims/builds/thor-cmu-201703101558-Linux64.zip
wget https://s3-us-west-2.amazonaws.com/ai2-vision-robosims/builds/thor-cmu-201703101557-OSXIntel64.zip
mkdir thor-cmu-201703101558-Linux64
unzip thor-cmu-201703101558-Linux64.zip -d thor-cmu-201703101558-Linux64
unzip thor-cmu-201703101557-OSXIntel64.zip
rm thor-cmu-201703101558-Linux64.zip
rm thor-cmu-201703101557-OSXIntel64.zip
cd ..
echo done.
