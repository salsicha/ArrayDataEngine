
from __future__ import annotations

import numpy as np
import os
import tiledb
import shutil
from .source import DataSources


# TODO: restful, self-descriptive API, like "print help"


class DataBuffer:
    """Buffer Class
    Attributes:
    Args:
    Returns:
    """


    def __init__(self, data_source: DataSources, buffer_depth=1, data_uri="/tmp/tiledb/my_group/", topics=[], axis="", use_db=False):
        """Constructor
        
        """
        # Set buffer depth as message count
        self.buffer_depth = buffer_depth
        self._axis = axis
        self.topics = topics
        self.use_db = use_db
        self._init_source = data_source
        self.group_uri = data_uri

        if not os.path.exists(self.group_uri):
            os.makedirs(self.group_uri, exist_ok=True)
            tiledb.group_create(self.group_uri)

        self.set_methods()

        self.reset()


    def get_group_uri(self) -> str:
        return self.group_uri


    def set_methods(self) -> None:

        # TODO: finish factory pattern? Pass arg to constructors?

        if not self.use_db:
            self.roll_buffer = self._roll_np_buffer
            self.append_buffer = self._append_np_buffer
            self.get_buffer = self._get_np_buffer
        else:
            self.roll_buffer = self._roll_tdb_buffer
            self.append_buffer = self._append_tdb_buffer
            self.get_buffer = self._get_tdb_buffer


    def reset(self) -> None:
        self.counters = {}
        self.timestamps = {}
        self.msg_len = {}
        self._data_buffer = {}

        # TODO: consistency problems with these global vars?

        data_source = self._init_source

        # If data_source is a function instead of a class, call it directly
        try:
            self.topics = data_source.get_topics()
            self.data_source = data_source.get_message()
        except:
            self.data_source = self._init_source()

        if not self._axis in self.topics:
            print(f"{self._axis} is not in {self.topics}")
        else:
            for i in range(self.buffer_depth):
                self.roll_buffer(self._axis)


    def reset_buffer(self):
        self.reset()
        self.roll_buffer(self._axis)


    def set_axis(self, axis: str) -> None:
        if not axis in self.topics:
            raise Exception(f"Axis: {axis} not in topics: {self.topics}")
        self._axis = axis


    def set_topics(self, topics):
        self.topics = topics


    def get_topics(self):
        return self.topics


    def get_size(self):
        return self.msg_len[self._axis]


    def load_data_db(self, axis: str) -> None:
        if not axis in self.topics:
            raise Exception(f"Axis: {axis} not in topics: {self.topics}")
        self._axis = axis
        while True:
            try:
                self.roll_buffer(self._axis)
            except Exception as e:
                print("Finished loading: ", str(e))

                if self.use_db:
                    for topic in self.topics:
                        uri = self._get_array_uri(topic)
                        with tiledb.open(uri, "w") as tiledb_array:
                            tiledb_array.meta["closed"] = True

                return


    # TODO: add methods to allow enumerate() to be called on this object
    def get_data(self, axis):
        if not axis in self.topics:
            raise Exception(f"Axis: {axis} not in topics: {self.topics}")
        self._axis = axis

        counter = 0
        while True:
            counter += 1
            try:
                self.roll_buffer(self._axis)
                yield self.get_buffer(), counter
            except StopIteration:
                print("End of source")
                self.reset_buffer()
                self.roll_buffer(self._axis)
                return


    def __getitem__(self, subscript: slice | int) -> np.ndarray | float | int:
        """
        The subscript is only over the first dimension
        """

        if not self.use_db:
            if isinstance(subscript, slice):
                return np.squeeze(self._data_buffer[self._axis]['data'][subscript.start, subscript.stop, subscript.step])
            elif isinstance(subscript, int):
                # Slice by time where float value represents seconds counting back
                return np.squeeze(self._data_buffer[self._axis]['data'][subscript])

                # TODO: slice by time index?
                # return self._data_buffer[self._axis]['data'][self._data_buffer[self._axis]['ts'] > \
                #                                             self._data_buffer[self._axis]['ts'][-1] - subscript]
            # else:
            #     return self._data_buffer[self._axis]['data'][subscript]

        else:
            if isinstance(subscript, slice):
                with tiledb.Group(self.group_uri) as g:
                    for a in g:
                        if a.name == self._axis:
                            with tiledb.DenseArray(a.uri) as A:
                                return A[subscript.start:subscript.stop]["features"]

            elif isinstance(subscript, int):
                # Since the TileDB array is initialized with zeros,
                # the last non-zero entry is (counter - 1)
                if subscript < 0:
                    subscript = self.counters[self._axis] + subscript

                with tiledb.Group(self.group_uri) as g:
                    for a in g:
                        if a.name == self._axis:
                            with tiledb.DenseArray(a.uri) as A:
                                return A[subscript]["features"]


    def __setitem__(self, subscript: slice | int, newval: np.ndarray) -> bool | None:
        """
        The subscript is only over the first dimension
        """

        if not self.use_db:
            if isinstance(subscript, slice):
                self._data_buffer[self._axis]['data'][subscript.start, subscript.stop, subscript.step] = newval
            elif isinstance(subscript, int):
                # Slice by time where float value represents seconds counting back
                self._data_buffer[self._axis]['data'][self._data_buffer[self._axis]['ts'] > \
                                                    self._data_buffer[self._axis]['ts'][-1] - subscript] = newval
            else:
                self._data_buffer[self._axis]['data'][subscript] = newval
        else:
            if isinstance(subscript, slice):
                with tiledb.Group(self.group_uri) as g:
                    for a in g:
                        if a.name == self._axis:
                            with tiledb.open(a.uri, "w") as A:
                                A[subscript.start:subscript.stop] = newval
                                return True

            elif isinstance(subscript, int):
                with tiledb.Group(self.group_uri) as g:
                    for a in g:
                        if a.name == self._axis:
                            with tiledb.open(a.uri, "w") as A:
                                A[subscript] = newval
                                return True


    def _get_tdb_buffer(self) -> dict:
        """Return buffer as dictionary
        """
        buffer = {}
        with tiledb.Group(self.group_uri) as g:
            for a in g:
                with tiledb.DenseArray(a.uri) as A:
                    if A.meta.get("topic") and A.meta.get("topic") in self.counters:
                        buffer[A.meta.get("topic")] = {}
                        buffer[A.meta.get("topic")]['id'] = A.meta.get("topic") # TODO: Is ID supposed to be the topic name?
                        buffer[A.meta.get("topic")]['ts'] = A.meta.get("timestamp")
                        buffer[A.meta.get("topic")]['data'] = A[0:self.counters[A.meta.get("topic")]]["features"]
        return buffer


    def _get_np_buffer(self) -> dict:
        """Return buffer as dictionary
        """

        return self._data_buffer.copy()


    def _get_array_uri(self, topic):
        uri = self.group_uri + topic.replace("/", "_")
        return uri


    def _init_tdb(self, msg: dict) -> None:
        data_len = self.msg_len[msg['topic']]

        uri = self._get_array_uri(msg['topic'])

        if os.path.exists(uri):
            with tiledb.open(uri, "r") as tiledb_array:
                try:
                    if tiledb_array.meta["closed"]:
                        print(f"Full data set exists! {uri}")
                        return
                except Exception as e:
                    print("Closed tag not present.")
            shutil.rmtree(uri)

        dims = [
            tiledb.Dim(
                name="images" if dim == 0 else "dim_" + str(dim - 1),
                domain=(0, data_len if dim == 0 else (msg['data'].shape[dim - 1] - 1)),
                tile=1 if dim == 0 else msg['data'].shape[dim - 1],
                dtype=np.int32,
            )
            for dim in range(msg['data'].ndim + 1)
        ]
        
        # TileDB schema
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(*dims),
            sparse=False,
            attrs=[tiledb.Attr(name="features", dtype=msg['data'].dtype)],
        )
        # Create array
        os.makedirs(uri, exist_ok=True)
        tiledb.Array.create(uri, schema)
    
        with tiledb.Group(self.group_uri, "w") as g:
            g.add(uri, msg['topic'])


    def _roll_tdb_buffer(self, axis: str) -> None:
        if not axis in self.topics:
            raise Exception(f"Axis: {axis} not in topics: {self.topics}")
        self._axis = axis

        while True:
            msg = next(self.data_source)
            
            ### TODO: 

            if not msg['topic'] in self.timestamps:
                self.timestamps[msg['topic']] = []
                self.counters[msg['topic']] = 0
                self.msg_len[msg['topic']] = self._init_source.get_count(msg['topic'])
                self._init_tdb(msg)

            # TODO: load existing data
            # if not os.path.exists(array_uri):
            #     self.timestamps[array_uri] = []
            #     self.counters[array_uri] = 0
            #     self.msg_len[array_uri] = self._init_source.get_count(msg['topic'])
            #     self._init_tdb(array_uri, msg['data'], self.msg_len[array_uri], msg['topic'])
            # else:
            #     with tiledb.open(array_uri) as tiledb_array:
            #         self.timestamps[array_uri] = tiledb_array.meta["timestamp"]
            #         self.counters[array_uri] = tiledb_array.meta["count"]

            self.append_buffer(msg)

            if msg['topic'] == self._axis:
                break


    def _roll_np_buffer(self, axis: str) -> None:
        if not axis in self.topics:
            raise Exception(f"Axis: {axis} not in topics: {self.topics}")
        self._axis = axis

        # Get message from source
        while True:
            msg = next(self.data_source)

            # Init numpy objects in buffer dict
            if not msg['topic'] in self._data_buffer:
                self._data_buffer[msg['topic']] = np.zeros(self.buffer_depth, dtype=[('ts', '<f8'), \
                                                                        ('id', 'S'), \
                                                                        ('data', msg['data'].dtype, msg['data'].shape)])

            self.append_buffer(msg)

            if msg['topic'] == self._axis:
                break


    def _append_np_buffer(self, msg: dict) -> None:

        # Sensor data
        self._data_buffer[msg['topic']]['data'][:-1] = self._data_buffer[msg['topic']]['data'][1:]
        self._data_buffer[msg['topic']]['data'][-1] = msg['data']

        # Timestamp
        self._data_buffer[msg['topic']]['ts'][:-1] = self._data_buffer[msg['topic']]['ts'][1:]
        self._data_buffer[msg['topic']]['ts'][-1] = msg['timestamp']

        # ID
        self._data_buffer[msg['topic']]['id'][:-1] = self._data_buffer[msg['topic']]['id'][1:]
        self._data_buffer[msg['topic']]['id'][-1] = msg['name']


    def _append_tdb_buffer(self, msg: dict) -> None:

        array_uri = self._get_array_uri(msg['topic'])

        if not os.path.exists(array_uri):
            self._init_tdb(msg)

        with tiledb.open(array_uri, "w") as tiledb_array:

            self.timestamps[msg['topic']].append(msg['timestamp'])

            tiledb_array[self.counters[msg['topic']], :] = msg['data']

            self.counters[msg['topic']] += 1

            # TODO: stop when ALL counters are full
            # if self.counters[array_uri] >= self.msg_len[array_uri]:

            tiledb_array.meta["timestamp"] = np.array(self.timestamps[msg['topic']])
            tiledb_array.meta["name"] = msg["name"]
            tiledb_array.meta["topic"] = msg["topic"]
            tiledb_array.meta["count"] = self.counters[msg['topic']]
