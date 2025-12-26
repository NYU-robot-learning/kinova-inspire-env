# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import time

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver

from projectaria_tools.core.sensor_data import (
    ImageData,
    ImageDataRecord,
)

# Set up the device client to initiate connection to the device
device_client = sdk_gen2.DeviceClient()


def device_streaming():
    # Set up the device client config to specify the device to be connected to e.g. device serial number.
    # If nothing is specified, the first device in the list of connected devices will be connected to
    config = sdk_gen2.DeviceClientConfig()
    device_client.set_client_config(config)
    device = device_client.connect()

    # Set recording config with profile name
    streaming_config = sdk_gen2.HttpStreamingConfig()
    streaming_config.profile_name = "profile9"
    streaming_config.streaming_interface = sdk_gen2.StreamingInterface.USB_NCM
    device.set_streaming_config(streaming_config)

    # Start and stop recording
    device.start_streaming()
    return device


def image_callback(image_data: ImageData, image_record: ImageDataRecord):
    print(
        f"Received image data of size {image_data.to_numpy_array().shape} with timestamp {image_record.capture_timestamp_ns} ns"
    )
    return image_data.to_numpy_array()


def setup_streaming_receiver(device):
    config = sdk_gen2.HttpServerConfig()
    config.address = "0.0.0.0"
    config.port = 6768

    # setup the receiver
    stream_receiver = receiver.StreamReceiver()
    stream_receiver.set_server_config(config)

    stream_receiver.register_rgb_callback(image_callback)

    # start the server
    stream_receiver.start_server()

    time.sleep(10)

    # stop streaming and terminate the server
    device.stop_streaming()

    time.sleep(2)



if __name__ == "__main__":
    # setup device to start streaming
    device = device_streaming()

    # setup streaming receiver to receive streaming data with callbacks
    setup_streaming_receiver(device)
