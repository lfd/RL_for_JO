SELECT * FROM keyword AS k, movie_keyword AS mk, char_name AS chn, cast_info AS ci WHERE chn.name NOT LIKE '%Sherlock%' AND (chn.name LIKE '%Tony%Stark%' OR chn.name LIKE '%Iron%Man%') AND k.keyword IN ('superhero', 'sequel', 'second-part', 'marvel-comics', 'based-on-comic', 'tv-special', 'fight', 'violence') AND mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;