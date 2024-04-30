cd /home/ec2-user

yum update -y
amazon-linux-extras install epel
yum groupinstall -y "Development Tools"
yum install -y \
    nc git amazon-cloudwatch-agent p7zip inotify-tools \
    zlib-devel bzip2 bzip2-devel readline-devel \
    sqlite sqlite-devel tk-devel libffi-devel xz-devel openssl11-devel
yum update -y

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -s -c ssm:us-east-2:AmazonCloudWatch-linux

sudo -u ec2-user -i <<'EOF'
git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
EOF

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /home/ec2-user/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /home/ec2-user/.bashrc
echo 'eval "$(pyenv init -)"' >> /home/ec2-user/.bashrc

sudo -u ec2-user -i <<'EOF'

pyenv install 3.11.0
pyenv global 3.11.0
pip install --upgrade pip
pip install scikit-learn numpy matplotlib pandas scipy skops
pip install git+https://github.com/hieulw/cicflowmeter

mkdir raw_traffic
mkdir discarded_traffic
mkdir models

cd assets
wget https://github.com/DiegoRealpe/DER_ML_ADS/raw/main/dataset/DNP3/Iowa_State_University_DER-DNP3_2022.7z
7za x Iowa_State_University_DER-DNP3_2022.7z -p'P0w34Cy&er#DER#DNP3' -o'.'
rm -rf Iowa_State_University_DER-DNP3_2022.7z
EOF

echo "Setup Complete"
