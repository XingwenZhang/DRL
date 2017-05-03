rm -rf thor_binary
mkdir -p thor_binary
cd thor_binary

# download OS-X build
wget --no-check-certificate https://s3-us-west-2.amazonaws.com/ai2-vision-robosims/builds/thor-201705011400-OSXIntel64.zip
unzip thor-201705011400-OSXIntel64.zip
rm thor-201705011400-OSXIntel64.zip

# download Linux build
wget --no-check-certificate https://s3-us-west-2.amazonaws.com/ai2-vision-robosims/builds/thor-201705011400-Linux64.zip
unzip thor-201705011400-Linux64.zip
rm thor-201705011400-Linux64.zip

# download Target Annotation
wget --no-check-certificate https://s3-us-west-2.amazonaws.com/ai2-vision-robosims/builds/thor-challenge-targets.zip
unzip thor-challenge-targets.zip
rm thor-challenge-targets.zip

cd ..
echo done.
