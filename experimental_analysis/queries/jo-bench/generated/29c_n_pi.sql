SELECT * FROM name AS n, person_info AS pi WHERE n.gender = 'f' AND n.name LIKE '%An%' AND n.id = pi.person_id AND pi.person_id = n.id;