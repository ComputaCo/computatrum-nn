import React, { useState, useEffect, useRef } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  TouchableWithoutFeedback,
  Keyboard,
} from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';
import axios from 'axios';

const Messages = ({ serverUrl, websocketUrl, onRefreshConnections }) => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const websocketRef = useRef(null);

  useEffect(() => {
    fetchMessages();

    websocketRef.current = new WebSocket(websocketUrl);
    websocketRef.current.onopen = () => {
      console.log('WebSocket connection opened');
    };
    websocketRef.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      setMessages(prevMessages => [...prevMessages, message]);
      onRefreshConnections();
    };
    websocketRef.current.onclose = () => {
      console.log('WebSocket connection closed');
    };

    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, []);

  const fetchMessages = async () => {
    try {
      const response = await axios.get(`${serverUrl}/messages`);
      setMessages(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  const sendMessage = async () => {
    if (inputText.trim() !== '') {
      try {
        const newMessage = {
          name: 'User', // Replace with user name or another identifier
          text: inputText,
          time: Date.now() / 1000,
        };

        await axios.post(`${serverUrl}/messages`, newMessage);
        setInputText('');
      } catch (error) {
        console.error(error);
      }
    }
  };

  const renderItem = ({ item }) => (
    <View style={styles.messageContainer}>
      <Text style={styles.messageName}>{item.name}:</Text>
      <Text style={styles.messageText}>{item.text}</Text>
      <DateTimePicker
        value={new Date(item.time * 1000)}
        mode="time"
        display="default"
        style={styles.messageTime}
        is24Hour={true}
        onChange={() => { }}
      />
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <KeyboardAvoidingView
          style={styles.container}
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        >
          <FlatList
            data={messages}
            renderItem={renderItem}
            keyExtractor={item => item.guid.toString()}
            onRefresh={fetchMessages}
          />
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              value={inputText}
              onChangeText={setInputText}
              placeholder="Type your message"
            />
            <TouchableOpacity onPress={sendMessage} style={styles.sendButton}>
              <Text style={styles.sendButtonText}>Send</Text>
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </TouchableWithoutFeedback>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  messageContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
  },
  messageName: {
    fontWeight: 'bold',
  },
  messageText: {
    marginLeft: 5,
  },
  messageTime: {
    marginLeft: 10,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
    padding: 5,
  },
  sendButton: {
    backgroundColor: '#3498db',
    borderRadius: 5,
    padding: 10,
    marginLeft: 10,
  },
  sendButtonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
});

export default Messages;