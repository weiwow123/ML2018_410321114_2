DATA=/home/cos/IML/Mnist_HandWriting/testing_data
echo "create testing.txt..."
rm -rf testing.txt
find $DATA -name *.png  >> testing.txt
echo "Done.."

