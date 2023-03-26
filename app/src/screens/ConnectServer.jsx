import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    StyleSheet,
    TouchableWithoutFeedback,
    Keyboard,
    KeyboardAvoidingView,
    Platform,
    Alert,
} from 'react-native';
import * as yup from 'yup';
import { useFormik } from 'formik';

const validationSchema = yup.object().shape({
    address: yup.string().trim().required('Please enter the server address.'),
    port: yup.string().trim().required('Please enter the port.'),
    websocketPort: yup.string().trim()
});

const ConnectServer = ({ onConnect, navigation }) => {

    const formik = useFormik({
        initialValues: {
            address: '192.168.86.38',
            port: '5000',
            websocketPort: '5001',
        },
        validationSchema,
        onSubmit: handleConnect,
    });

    const { handleChange, handleBlur, handleSubmit, values, errors, touched } = formik;

    useEffect(() => {
        if (values && values.port && !values.websocketPort) {
            const websocketPort = Number(values.port) + 1;
            console.log('Setting websocket port to', websocketPort)
            handleChange('websocketPort')(websocketPort);
        }
    }, [values.port]);

    async function testConnection(url) {
        try {
            const response = await fetch(url, { method: 'HEAD', timeout: 2000 });
            return response.ok;
        } catch (error) {
            console.error('Connection test failed:', error);
            return false;
        }
    }

    async function handleConnect() {
        const url = `http://${values.address}:${values.port}`;
        const connectionSuccessful = await testConnection(url);

        if (connectionSuccessful) {
            onConnect(url, `ws://${values.address}:${values.websocketPort}`);
            navigation.navigate('Connections', {});
        } else {
            Alert.alert('Connection Failed', 'Unable to connect to the server. Please check the server address and port.');
        }
    }

    return (
        <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
            <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
                <View style={styles.content}>
                    <Text style={styles.title}>Connect to Server</Text>
                    <TextInput
                        style={styles.input}
                        value={values.address}
                        onChangeText={handleChange('address')}
                        onBlur={handleBlur('address')}
                        placeholder="Server Address"
                        keyboardType="url"
                        autoCapitalize="none"
                    />
                    {errors.address && touched.address && <Text style={styles.errorText}>{errors.address}</Text>}
                    <TextInput
                        style={styles.input}
                        value={values.port}
                        onChangeText={handleChange('port')}
                        onBlur={handleBlur('port')}
                        placeholder="Server Port"
                        keyboardType="numeric"
                    />
                    {errors.port && touched.port && <Text style={styles.errorText}>{errors.port}</Text>}
                    <TextInput
                        style={styles.input}
                        value={values.websocketPort}
                        onChangeText={handleChange('websocketPort')}
                        onBlur={handleBlur('websocketPort')}
                        placeholder={values.websocketPort || (values.port && `${Number(values.port) + 1}`) || 'WebSocket Port'}
                        keyboardType="numeric"
                    />
                    {errors.websocketPort && touched.websocketPort && <Text style={styles.errorText}>{errors.websocketPort}</Text>}
                    <TouchableOpacity onPress={handleSubmit} style={styles.connectButton}>
                        <Text style={styles.connectButtonText}>Connect</Text>
                    </TouchableOpacity>
                </View>
            </KeyboardAvoidingView>
        </TouchableWithoutFeedback>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
    },
    content: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    title: {
        fontSize: 24,
        marginBottom: 30,
    },
    input: {
        borderWidth: 1,
        borderColor: '#ccc',
        borderRadius: 5,
        width: '80%',
        padding: 5,
        marginBottom: 15,
    },
    connectButton: {
        backgroundColor: '#3498db',
        borderRadius: 5,
        padding: 10,
    },
    connectButtonText: {
        color: '#fff',
        fontWeight: 'bold',
    },
});

export default ConnectServer;
