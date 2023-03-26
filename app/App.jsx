import React, { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

import ConnectServer from './src/screens/ConnectServer';
import Connections from './src/screens/Connections';
import Messages from './src/screens/Messages';

const Stack = createStackNavigator();

const App = () => {
  const [connections, setConnections] = useState([]);

  const handleConnect = (serverUrl, websocketUrl) => {
    const connection = { serverUrl, websocketUrl };
    if (!connections.some(conn => conn.serverUrl === serverUrl)) {
      setConnections([...connections, connection]);
      console.log('connections', connections);
    }
  };

  const refreshConnections = () => {
    setConnections([...connections]);
    console.log('connections', connections);
  };

  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="ConnectServer">
        <Stack.Screen name="ConnectServer">
          {props => <ConnectServer {...props} onConnect={handleConnect} navigation={props.navigation} />}
        </Stack.Screen>
        <Stack.Screen name="Connections">
          {props => (
            <Connections {...props}
              connections={connections}
              onSelectConnection={(serverUrl, websocketUrl) => {
                props.navigation.navigate('MessageList', { serverUrl, websocketUrl });
              }} />
          )}
        </Stack.Screen>
        <Stack.Screen name="MessageList">
          {props => (
            <Messages {...props} onRefreshConnections={refreshConnections} />
          )}
        </Stack.Screen>
      </Stack.Navigator>
      <StatusBar style="auto" />
    </NavigationContainer>
  );
};

export default App;
