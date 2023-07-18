# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""tvm.contrib.msc.core.utils.namespace"""

import copy


class MSC_MAP:
    """Global Namespace map for MSC"""

    MAP = {}

    @classmethod
    def set(cls, key, value):
        cls.MAP[key] = value

    @classmethod
    def get(cls, key, default=None):
        return cls.MAP.get(key, default)

    @classmethod
    def clone(cls, key, default=None):
        return copy.deepcopy(cls.get(key, default))

    @classmethod
    def delete(cls, key):
        if key in cls.MAP:
            return cls.MAP.pop(key)
        return None

    @classmethod
    def contains(cls, key):
        return key in cls.MAP


class MSC_KEY:
    """Keys for the MSC_MAP"""

    WORKSPACE = "workspace"
    VERBOSE = "verbose"
